import numpy as np
from ultralytics import YOLO
import bettercam
import win32con

from main_version import (
    MOUSEINPUT, INPUT, move_mouse,
    get_cursor_position, is_key_pressed
)

EXIT_KEY = win32con.VK_F2
CONFIDENCE_THRESHOLD = 0.5  # Sets the minimum confidence threshold for detections
TARGET_FPS = 80
TARGET_CLASS = 't'  # Chose class 'ct' or 't'

# Load your YOLO model
model = YOLO("models_fp16/LowFP16.pt").to('cuda')

camera = bettercam.create(max_buffer_len=1)

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001


def main():
    print(f"Script started.")
    print(f"Press F2 to EXIT.")

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
                for idx, cls in enumerate(result.boxes.cls):
                    if names[int(cls)] == TARGET_CLASS:
                        coordinates = result.boxes.xywh.cpu().numpy()[idx]
                        x, y, w, h = coordinates[:4]
                        center_x = int(x)
                        center_y = int(y)

                        current_x, current_y = get_cursor_position()

                        dx = int(center_x - current_x)
                        dy = int(center_y - current_y - h / 2.5)

                        # Adjust the factors as needed to fine-tune the responsiveness
                        min_height = 10  # Minimum height of object (close)
                        max_height = 200  # Maximum height of object (far)
                        min_speed = 0.9  # Minimum speed factor (far)
                        max_speed = 1  # Maximum speed factor (close)

                        clamped_h = np.clip(h, min_height, max_height)
                        distance_factor = (max_height - clamped_h) / \
                            (max_height - min_height)
                        speed_factor = min_speed + \
                            (max_speed - min_speed) * (1 - distance_factor)

                        move_mouse(dx, dy, speed_factor)

                        break

    except Exception as e:
        print(f'An error occurred: {e}')
    finally:
        camera.stop()
        print("Camera stopped.")


if __name__ == '__main__':
    main()
