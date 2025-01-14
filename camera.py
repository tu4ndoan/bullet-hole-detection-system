import cv2
import os
import logging
from threading import Lock
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from PIL import Image, ImageTk


cameras = {}
targets = []
class Camera:
    def __init__(self, camera_id, lane=None, target=None):
        self.camera_id = camera_id
        self.lane = lane
        self.target = target
        self.cap = None
        self.lock = Lock() # To ensure thread safety when accessing the camera
        self.is_active = True  

    def activate(self):
        self.is_active = True
        print(f"Camera {self.camera_id} activated")

    def deactivate(self):
        self.is_active = False
        print(f"Camera {self.camera_id} deactivated due to connection loss")

    def capture_image(self, turn):
        #if not self.is_active:
        #    print(f"Camera {self.camera_id} bệ số {self.lane} mục tiêu {self.target} mất kết nổi, hãy kiểm tra lại và bấm quét camera")
        with self.lock:  # Ensuring only one thread can access the camera at a time
            try:
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open camera {self.camera_id}")

                logging.info(f"Camera {self.camera_id} initialized successfully.")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

                # Wait a bit to ensure the camera stabilizes
                cv2.waitKey(2000)
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError("Failed to capture frame")

                # Save and display image
                self.save_image(frame, turn)
                return frame
            except Exception as e:
                logging.error(f"Error capturing image from camera {self.camera_id}: {e}")
                return None
            finally:
                if self.cap:
                    self.release()

    def save_image(self, frame, turn):
        path = f"./HinhAnh/DaiBan{self.lane}/"
        if not os.path.exists(path):
            os.makedirs(path)

        image_path = f"{path}{self.target}-{self.lane}-{turn}.jpg"
        try:
            cv2.imwrite(image_path, frame)
            logging.info(f"Image saved to {image_path}")
        except Exception as e:
            logging.error(f"Error saving image: {e}")

    def release(self):
        if self.cap:
            print("released")
            self.cap.release()
            logging.info(f"Released camera {self.camera_id}")
        else:
            print("cant release")
        cv2.destroyAllWindows()

    def set_lane(self, lane):
        self.lane = lane

    def set_target(self, target):
        self.target = target

    def get_lane(self):
        return self.lane

    def get_target(self):
        return self.target


# Function to create and store the Camera object based on user input
def create_camera_object(camera_id, lane, target):
    print(camera_id,lane, target)
    # Validate the input for target and lane
    if not target or not lane:
        messagebox.showerror("Lỗi", f"Hãy nhập đủ các thông tin tên bia, dải bắn của camera {camera_id}")
        return
    if camera_id in cameras:
        print("exist")
        cameras[camera_id].activate()
        logging.info(f"Camera {camera_id} already exists.")
        return

    # Create a new camera object
    camera_obj = Camera(camera_id, lane, target)
    cameras[camera_id] = camera_obj  # Add the new camera to the dictionary
    logging.info(f"Camera {camera_id} {target} created and added to the dictionary.")
    logging.info(f"Current cameras dictionary: {cameras}")
    camera_obj.release()
    if (target not in targets):
        targets.append(target)
    messagebox.showinfo("Thông báo",f"Đã nhập 1 camera: \nDải số: {lane} \nMục tiêu: {target}")
    
    # Close the Toplevel window after applying
    editor_window.destroy()

# Function to open the variable input window
# Function to show a mini image from the camera in the editor window
def show_mini_image(camera_id, editor_window):
    # Capture image from the camera
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
    
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    # Resize the image to a smaller size for display
    image = image.resize((150, 150))  # Resize to 150x150 for a mini image

    # Convert PIL image to Tkinter format
    photo = ImageTk.PhotoImage(image)

    # Display the image in the Tkinter window
    label = ttk.Label(editor_window, image=photo)
    label.image = photo  # Keep a reference to avoid garbage collection
    label.pack(pady=5)

# Function to open the variable editor for a specific camera
def open_variable_editor(camera_id):
    if camera_id in cameras:
        return  # Camera already exists, no need to reopen

    global editor_window
    editor_window = tk.Toplevel()
    editor_window.title(f"Variable Editor for Camera {camera_id}")
    # Show mini image from the camera
    show_mini_image(camera_id, editor_window)
    # Create the camera object

    # Create a variable to store the selected value for the target (bia)
    bia_var = tk.StringVar()
    bia_var.set("BiaSo4")
    
    # Create the option menu with predefined values for bia
    bia_options = ["BiaSo4", "BiaSo7"]
    bia_menu = tk.OptionMenu(editor_window, bia_var, *bia_options)
    bia_menu.pack(pady=5)

    ttk.Label(editor_window, text="Dải số:").pack(pady=5)

    # Create an entry widget for entering the lane number
    lane_entry = ttk.Entry(editor_window)
    lane_entry.insert(0, "1")  # Default value for lane
    lane_entry.pack(pady=5)

    # Create the "Apply and Close" button
    apply_button = ttk.Button(
        editor_window, 
        text="OK", 
        command=lambda camera_id=camera_id: create_camera_object(camera_id, lane_entry.get(), bia_var.get())
    )
    apply_button.pack(pady=20)

    # Wait until the editor window is closed
    editor_window.protocol("WM_DELETE_WINDOW", editor_window.destroy)
    editor_window.wait_window(editor_window)


def view_camera(camera_id):
    try:
        cam = cameras[camera_id]
        img = cam.capture_image(99)
        cv2.imshow(f"camera {camera_id} - dai so {cam.get_lane()} - muc tieu {cam.get_target()}",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)

def view_all_camera():
    new_window = tk.Toplevel()
    new_window.title("Tất cả camera")
    new_window.geometry("400x300")
    for camera_id in cameras:
        if not cameras[camera_id].is_active:
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.release()
                cameras[camera_id].activate()
        if cameras[camera_id].is_active:
            camera_btn = tk.Button(new_window, text=f"camera {camera_id}", command=lambda camera_id=camera_id: view_camera(camera_id))
            camera_btn.pack(pady=5)
    print("View all connected and registered cameras")

def parallel_capture(turn):
    
    threads = []

    # Create threads for each camera
    for camera_id in cameras:
        t = threading.Thread(target=cameras[camera_id].capture_image, args=(turn,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    cv2.destroyAllWindows()
