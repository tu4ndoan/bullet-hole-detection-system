import cv2
import threading
import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

targets = []
camera_objects = []
class Camera:
    def __init__(self, lane=None, target=None, camera_id = 0):
        self.cap = None
        self.lane = lane  # Lane the camera is focused on
        self.target = target  # Target object or area the camera is aimed at
        self.camera_id = camera_id

    def capture_image(self, turn):
        # Capture a single frame from the camera
        # Initialize the camera object
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)  # Open the default camera
        # Wait for the webcam to initialize
        while not self.cap.isOpened():
            print(f"Waiting for camera {self.camera_id} to initialize...")
            cv2.waitKey(100)  # Wait for 100 ms before checking again

        print(f"Camera {self.camera_id} initialized successfully!")
        # Set resolution to 1080p
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Verify if the resolution is set correctly
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Resolution: {width}x{height}")
        cv2.waitKey(2000) # wait 2 sec so the lighting is stable
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame")
        else:
            if frame is None or frame.size == 0:
                print("Error: Captured frame empty")
            else:
                self.save_image(frame, turn)
                
            return frame
        
    def show_image(self, frame):
        # Display the captured frame in a window
        cv2.imshow("Captured Image", frame)

    def save_image(self, frame, turn):
        path = f"./HinhAnh/DaiBan{self.lane}/"
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(f"{path}{self.target}-{self.lane}-{turn}.jpg", frame)

    def set_lane(self, lane):
        # Set the lane property
        self.lane = lane

    def set_target(self, target):
        # Set the target property
        self.target = target

    def get_lane(self):
        # Get the lane property
        return self.lane

    def get_target(self):
        # Get the target property
        return self.target

    def get_camera_id(self):
        return self.camera_id

    def release(self):
        # Release the camera and close any OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

# Function to create and store the Camera object based on user input
def create_camera_object(camera_id, lane, target):
    top
    # Validate the input for target and lane
    if not target or not lane:
        messagebox.showerror("Lỗi", f"Hãy nhập đủ các thông tin tên bia, dải bắn của camera {camera_id}")
        return

    # Create a Camera object with the values entered
    camera_obj = Camera(lane, target, camera_id)
    if camera_obj not in camera_objects:
        camera_objects.append(camera_obj)
    if (target not in targets):
        targets.append(target)
    messagebox.showinfo("Thông báo",f"Đã nhập 1 camera: \nDải số: {lane} \nMục tiêu: {target}")
    
    # Close the Toplevel window after applying
    top.destroy()

# Function to open the variable input window
def open_variable_editor(camera_id):
    global top
    # Create a variable to store the selected value
    bia_var = tk.StringVar()
    # Set a default value for the option menu
    bia_var.set("BiaSo4")
    
    # Create a new Toplevel window
    top = tk.Toplevel()
    top.title(f"Camera số {camera_id}")

    # Create the option menu with the predefined values
    bia_options = ["BiaSo4", "BiaSo7", "BiaSo10"]
    bia_menu = tk.OptionMenu(top, bia_var, *bia_options)
    bia_menu.pack(pady=5)
    ttk.Label(top, text="Dải số:").pack(pady=5)
    lane_entry = ttk.Entry(top)
    lane_entry.insert(0, "1")  # Default value
    lane_entry.pack(pady=5)
    
    # Create the "Apply and Close" button
    apply_button = ttk.Button(top, text="OK", command=lambda: create_camera_object(camera_id, lane_entry.get(), bia_var.get()))
    apply_button.pack(pady=20)

    top.geometry("300x200")

def view_camera(camera_id):
    print(camera_id)
    try:
        cam = camera_objects[camera_id-1]
        img = cam.capture_image(0)
        cv2.imshow(f"camera {camera_id} - dai so {cam.get_lane()} - muc tieu {cam.get_target()}",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)

def view_all_camera():
    new_window = tk.Toplevel()
    new_window.title("Tất cả camera")
    new_window.geometry("400x300")
    for camera in camera_objects:
        camera_id = camera.get_camera_id()
        camera_btn = tk.Button(new_window, text=f"camera {camera_id}", command=lambda: view_camera(camera_id))
        camera_btn.pack(pady=5)
    print("View all connected and registered cameras")

def parallel_capture(turn):
    
    threads = []

    # Create threads for each camera
    for camera in camera_objects:
        t = threading.Thread(target=camera.capture_image, args=(turn,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    cv2.destroyAllWindows()