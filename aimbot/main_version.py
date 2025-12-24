from ultralytics import YOLO
import win32api
import ctypes
import bettercam
import numpy as np
import win32con


EXIT_KEY = win32con.VK_F2
CONFIDENCE_THRESHOLD = 0.5  # Sets the minimum confidence threshold for detections
TARGET_FPS = 80


# Load your YOLO model
model = YOLO("models_fp16/LowFP16.pt").to('cuda')

camera = bettercam.create(max_buffer_len=1)

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT)
    ]


def move_mouse(dx, dy, speed_factor=1):
    extra = ctypes.c_ulong(0)
    input_ = INPUT()
    input_.type = INPUT_MOUSE

    scaled_dx = int(dx * speed_factor)
    scaled_dy = int(dy * speed_factor)

    if scaled_dx == 0 and scaled_dy == 0:
        return

    input_.mi = MOUSEINPUT(scaled_dx, scaled_dy, 0,
                           MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))

    ctypes.windll.user32.SendInput(
        1, ctypes.byref(input_), ctypes.sizeof(input_))


def get_cursor_position():
    return win32api.GetCursorPos()


def is_key_pressed(key_code):
    return win32api.GetAsyncKeyState(key_code) & 0x8000


def main():
    print(f"Script started.")
    print(f"Press F2 to EXIT.")

    camera.start(target_fps=TARGET_FPS)

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
                if len(boxes) > 0:
                    coordinates = boxes.xywh.cpu().numpy()

                    x, y, w, h = coordinates[0][:4]
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
