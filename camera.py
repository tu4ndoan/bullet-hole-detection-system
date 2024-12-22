import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

camera_indices = [0]  # Add more indices if you have more cameras connected
num_cameras = len(camera_indices)

class CameraApp:
    def __init__(self, root, tab, camera_ids=[0, 1]):
        self.root = root
        self.camera_ids = camera_ids
        self.tab = tab  # Use an existing tab passed as an argument
        self.cap = []  # To store VideoCapture objects for each camera
        self.labels = []  # To store labels for each camera
        self.frames = []  # To store frames of each camera feed

        # Create a frame within the existing tab to hold the labels for multiple camera feeds
        self.camera_frame = tk.Frame(self.tab)
        self.camera_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Create a label for each camera feed
        for _ in self.camera_ids:
            label = tk.Label(self.camera_frame)
            label.pack(side="left", padx=10, pady=10)
            self.labels.append(label)

        # Initialize the cameras
        for camera_id in self.camera_ids:
            self.cap.append(cv2.VideoCapture(camera_id))
            if not self.cap[-1].isOpened():
                print(f"Error: Unable to open camera with ID {camera_id}.")
        
        # Start updating the camera feeds
        self.update_frames()

    def update_frames(self):
        """ Capture frames from each camera and update the respective labels. """
        for i, cap in enumerate(self.cap):
            ret, frame = cap.read()
            if ret:
                # Convert the frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert the frame to a PhotoImage object
                image = Image.fromarray(frame_rgb)
                tk_image = ImageTk.PhotoImage(image)

                # Update the corresponding label with the new image
                self.labels[i].configure(image=tk_image)
                self.labels[i].image = tk_image  # Keep a reference to avoid garbage collection

        # Call the update_frames method again after 10 ms to refresh the frames
        self.root.after(10, self.update_frames)

    def __del__(self):
        """ Release the cameras when the application is closed. """
        for cap in self.cap:
            if cap.isOpened():
                cap.release()

# Function to handle mouse click event
def on_mouse_click(event, x, y, flags, param):
    newwin = tk.Tk()
    newwin.geometry("800x600")
    global camera_indices
    if event == cv2.EVENT_LBUTTONDOWN:
        # Calculate the index of the selected camera based on mouse position
        grid_width = 640  # Width of each camera feed in the grid
        grid_height = 480  # Height of each camera feed in the grid
        
        # Determine which camera was clicked by comparing the x, y position
        camera_index = (y // grid_height) * 2 + (x // grid_width)  # Assuming 2 cameras per row
        
        new_window = param
        # Create labels and entry fields for 3 parameters
        label1 = tk.Label(new_window, text="Dải bắn sô:")
        label1.pack(pady=5)
        shooting_lane = tk.Entry(new_window)
        shooting_lane.pack(pady=5)

        label2 = tk.Label(new_window, text="Mục tiêu:")
        label2.pack(pady=5)
        target_name = tk.Entry(new_window)
        target_name.pack(pady=5)

        label3 = tk.Label(new_window, text="Camera Id:")
        label3.pack(pady=5)
        camera_id = tk.Entry(new_window)
        camera_id.pack(pady=5)

        # Create a Submit button that calls on_submit with the entered parameters
        submit_button = tk.Button(new_window, text="Thêm", command=lambda: on_submit(shooting_lane.get(), target_name.get(), camera_id.get(), new_window))
        submit_button.pack(pady=20)

        if camera_index < len(camera_indices):
            print(f"Selected Camera {camera_indices[camera_index]} at position: ({x}, {y})")
            process_camera(new_window, camera_indices[camera_index])  # Process the selected camera

# Function to process selected camera feed
def process_camera(new_window, camera_index):
    # Create labels and entry fields for 3 parameters
    label1 = tk.Label(new_window, text="Dải bắn sô:")
    label1.pack(pady=5)
    shooting_lane = tk.Entry(new_window)
    shooting_lane.pack(pady=5)

    label2 = tk.Label(new_window, text="Mục tiêu:")
    label2.pack(pady=5)
    target_name = tk.Entry(new_window)
    target_name.pack(pady=5)

    # Create a Submit button that calls on_submit with the entered parameters
    submit_button = tk.Button(new_window, text="Thêm", command=lambda: on_submit(shooting_lane.get(), target_name.get(), camera_index, new_window))
    submit_button.pack(pady=20)

def add_camera(lane, target, camera_id):
    print(f"added 1 camera {camera_id}")   

def on_submit(param1, param2, param3, window):
    print(f"Parameters received: {param1}, {param2}, {param3}")
    
    # Call a function with the parameters (example function)
    result = add_camera(param1, param2, param3)
    
    # Close the new window after submitting
    window.destroy()
# Function to capture and display multiple camera feeds in one window
def display_all_cameras(root):
    # Open all cameras
    caps = [cv2.VideoCapture(i) for i in camera_indices]
    
    # Check if each camera is opened successfully
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_indices[i]}")
            caps[i].release()
            return  # Exit if camera cannot be opened

    while True:
        # Capture frames from all cameras
        frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))  # Blank frame if failed to capture

        # Resize frames to fit in a grid (e.g., 2x2 grid)
        resized_frames = []
        for i, frame in enumerate(frames):
            resized_frame = cv2.resize(frame, (640, 480))  # Resize each frame to fit the grid
            resized_frames.append(resized_frame)

        # Stack frames into a single image (grid layout)
        grid_image = np.vstack([
            np.hstack(resized_frames[i:i + 2]) for i in range(0, len(resized_frames), 2)
        ])

        # Display the grid of camera feeds
        cv2.imshow('All Camera Feeds', grid_image)
        
        # Set mouse callback to handle click events on the grid
        cv2.setMouseCallback('All Camera Feeds', on_mouse_click, param=root)

        # Wait for key press
        key = cv2.waitKey(1)
        if key == 27:  # Escape key to exit
            break

    # Release the camera resources
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

