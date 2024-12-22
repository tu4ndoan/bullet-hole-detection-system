import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import threading

class CameraApp:
    def __init__(self, root, camera_id):
        print("init")
        self.root = root
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        # Start the camera feed in a separate thread
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.daemon = True
        self.thread.start()

    def update_frame(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                print("ret")
                # Convert frame to RGB (OpenCV uses BGR by default)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = ImageTk.PhotoImage(img)

                # Update canvas with the new frame
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.canvas.image = img

    def stop(self):
        self.cap.release()

# Function to display the camera feed in a new window
def display_camera(camera_id, root):
    new_window = tk.Toplevel(root)
    new_window.title(f"Camera {camera_id}")
    new_window.geometry("800x600")

    app = CameraApp(new_window, camera_id)

# Example of how to pass camera_id and show camera feed
def add_camera(lane, target, camera_id, tab, root):
    print(f"Added 1 camera {camera_id}")
    camera_btn = tk.Button(tab, text=f"{target}-{lane}-{camera_id}", command=lambda: display_camera(camera_id,root))
    camera_btn.pack(pady=5)

# Assuming you already have other code for creating Tkinter interface and handling events
