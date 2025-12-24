import numpy as np
from ultralytics import YOLO
import bettercam
import win32con
from multiprocessing import Process


from main_version import (
    MOUSEINPUT, INPUT, move_mouse,
    get_cursor_position, is_key_pressed,
    run_overlay, get_screen_resolution
)

EXIT_KEY = win32con.VK_F2
CONFIDENCE_THRESHOLD = 0.5  # Sets the minimum confidence threshold for detections
TARGET_FPS = 80
TARGET_CLASS = 't'  # Chose class 'ct' or 't'
FOV_RADIUS = 150  # radius of the circle in pixels
DRAW_FOV = True

# Load your YOLO model
model = YOLO("models_fp16/LowFP16.pt").to('cuda')

camera = bettercam.create(max_buffer_len=1)

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001


def main():
    print(f"Script started.")
    print(f"Press F2 to EXIT.")

    screen_w, screen_h = get_screen_resolution()
    center_screen_x = screen_w // 2
    center_screen_y = screen_h // 2

    overlay_process = None
    if DRAW_FOV:
        overlay_process = Process(target=run_overlay, args=(FOV_RADIUS,))
        overlay_process.start()
        print("Overlay started.")

    camera.start(target_fps=TARGET_FPS)

    names = model.names

    try:
        while True:
            if is_key_pressed(EXIT_KEY):
                print("Exit key pressed. Stopping...")
                break

            screen = camera.get_latest_frame()

            results = model(screen, conf=CONFIDENCE_THRESHOLD,
                            stream=True, verbose=False)

            for result in results:
                boxes = result.boxes

                if not boxes:
                    break

                coordinates_array = boxes.xywh.cpu().numpy()
                classes_array = boxes.cls.cpu().numpy()

                closest_dist = float('inf')
                target_dx = 0
                target_dy = 0
                target_h = 0
                found_target = False

                current_x, current_y = get_cursor_position()

                for idx, cls in enumerate(classes_array):
                    if names[int(cls)] == TARGET_CLASS:
                        x, y, w, h = coordinates_array[idx][:4]
                        target_x = int(x)
                        target_y = int(y)

                        move_dx = int(target_x - current_x)
                        move_dy = int(target_y - current_y - h / 2.5)

                        fov_dx = target_x - center_screen_x
                        fov_dy = (target_y - h / 2.5) - center_screen_y

                        dist_from_center = np.hypot(fov_dx, fov_dy)

                        if dist_from_center < FOV_RADIUS and dist_from_center < closest_dist:
                            closest_dist = dist_from_center
                            target_dx = move_dx
                            target_dy = move_dy
                            target_h = h
                            found_target = True

                if found_target:
                    # Adjust the factors as needed to fine-tune the responsiveness
                    min_height = 10  # Minimum height of object (close)
                    max_height = 200  # Maximum height of object (far)
                    min_speed = 0.9  # Minimum speed factor (far)
                    max_speed = 1  # Maximum speed factor (close)

                    clamped_h = np.clip(target_h, min_height, max_height)
                    distance_factor = (max_height - clamped_h) / \
                        (max_height - min_height)
                    speed_factor = min_speed + \
                        (max_speed - min_speed) * (1 - distance_factor)

                    move_mouse(target_dx, target_dy, speed_factor)

                break

    except Exception as e:
        print(f'An error occurred: {e}')
    finally:
        camera.stop()
        if overlay_process and overlay_process.is_alive():
            overlay_process.terminate()
        print("Camera stopped.")


if __name__ == '__main__':
    main()
