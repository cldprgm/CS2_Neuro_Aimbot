from ultralytics import YOLO
import win32api
import ctypes
import bettercam
import numpy as np
import win32con
import win32gui
from multiprocessing import Process
import tkinter as tk


try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()


EXIT_KEY = win32con.VK_F2
CONFIDENCE_THRESHOLD = 0.5  # Sets the minimum confidence threshold for detections
FOV_RADIUS = 150  # radius of the circle in pixels
DRAW_FOV = True
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
    if dx == 0 and dy == 0:
        return
    extra = ctypes.c_ulong(0)
    input_ = INPUT()
    input_.type = INPUT_MOUSE

    scaled_dx = int(dx * speed_factor)
    scaled_dy = int(dy * speed_factor)

    input_.mi = MOUSEINPUT(scaled_dx, scaled_dy, 0,
                           MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))

    ctypes.windll.user32.SendInput(
        1, ctypes.byref(input_), ctypes.sizeof(input_))


def get_cursor_position():
    return win32api.GetCursorPos()


def get_screen_resolution():
    return win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)


def is_key_pressed(key_code):
    return win32api.GetAsyncKeyState(key_code) & 0x8000


def run_overlay(fov_radius):
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = tk.Tk()
    root.title("Overlay")

    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()

    root.geometry(f"{w}x{h}+0+0")
    root.overrideredirect(True)
    root.wm_attributes("-topmost", True)
    root.wm_attributes("-transparentcolor", "black")

    canvas = tk.Canvas(root, width=w, height=h,
                       bg="black", highlightthickness=0)
    canvas.pack()

    center_x, center_y = w // 2, h // 2

    canvas.create_oval(center_x - fov_radius, center_y - fov_radius,
                       center_x + fov_radius, center_y + fov_radius,
                       outline="green", width=2)

    hwnd = win32gui.FindWindow(None, "Overlay")
    ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style |
                           win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)

    root.mainloop()


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

                closest_dist = float('inf')
                target_dx = 0
                target_dy = 0
                target_h = 0
                found_target = False

                current_x, current_y = get_cursor_position()

                for box in coordinates_array:
                    x, y, w, h = box[:4]
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
                    distance_factor = (
                        max_height - clamped_h) / (max_height - min_height)
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
