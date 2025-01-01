import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import os
import camera
import image_processing
import cv2

# Create main window
root = tk.Tk()
root.title("TIỂU ĐOÀN 1038 - PHẦN MỀM BÁO BIA TỰ ĐỘNG")
root.geometry("800x600")

# Create a Notebook widget to hold tabs
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)
label = ttk.Frame(notebook)

# Variables
num_lane = len(notebook.tabs())
num_turn = 1

# Image preprocessing variables
blur_value = 1
adaptive_thresh_value = 1
binary_thresh_value = 1
edge_lower_value = 1
edge_higher_value = 1

# Hough circles variables
dp_value = 1
min_dist_value = 1
param1 = 1
param2 = 1
min_radius = 1
max_radius = 1
camera_to_target_distance = 1

# Dictionary to store detected cameras
camera_detected = {}
camera_objects = []
targets = [] #lets make the user input this

# Function to update global variables
def update_variables():
    global blur_value, adaptive_thresh_value, binary_thresh_value, edge_lower_value, edge_higher_value
    global dp_value, min_dist_value, param1, param2, min_radius, max_radius, camera_to_target_distance
    
    blur_value = blur_slider.get()
    adaptive_thresh_value = adaptive_thresh_slider.get()
    binary_thresh_value = binary_thresh_slider.get()
    edge_lower_value = edge_lower_slider.get()
    edge_higher_value = edge_higher_slider.get()
    
    dp_value = dp_slider.get()
    min_dist_value = min_dist_slider.get()
    param1 = param1_slider.get()
    param2 = param2_slider.get()
    min_radius = min_radius_slider.get()
    max_radius = max_radius_slider.get()
    camera_to_target_distance = camera_distance_slider.get()
    
    # Update the result label to display the updated values
    result_label.config(text=f"Updated Values:\nBlur: {blur_value}\nAdaptive Threshold: {adaptive_thresh_value}\nBinary Threshold: {binary_thresh_value}\n"
                            f"Edge Lower: {edge_lower_value}\nEdge Higher: {edge_higher_value}\n\n"
                            f"DP: {dp_value}\nMin Dist: {min_dist_value}\nParam1: {param1}\nParam2: {param2}\n"
                            f"Min Radius: {min_radius}\nMax Radius: {max_radius}\nCamera Distance: {camera_to_target_distance}")

# Create a new Toplevel window to edit variables
def open_variable_editor():
    global blur_slider, adaptive_thresh_slider, binary_thresh_slider, edge_lower_slider, edge_higher_slider
    global dp_slider, min_dist_slider, param1_slider, param2_slider, min_radius_slider, max_radius_slider, camera_distance_slider, result_label
    
    # Create a new Toplevel window
    top = tk.Toplevel()
    top.title("Global Variables Editor")
    
    # Create a canvas widget
    canvas = tk.Canvas(top)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create a scrollbar widget
    scrollbar = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill="y")

    # Configure the canvas to work with the scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Create a frame inside the canvas to contain all widgets
    canvas_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=canvas_frame, anchor="nw")

    # Add all the sliders (trackbars) inside this frame
    ttk.Label(canvas_frame, text="Blur Value:").pack(pady=5)
    blur_slider = tk.Scale(canvas_frame, from_=0, to=20, orient="horizontal")
    blur_slider.set(blur_value)
    blur_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Adaptive Threshold Value:").pack(pady=5)
    adaptive_thresh_slider = tk.Scale(canvas_frame, from_=0, to=20, orient="horizontal")
    adaptive_thresh_slider.set(adaptive_thresh_value)
    adaptive_thresh_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Binary Threshold Value:").pack(pady=5)
    binary_thresh_slider = tk.Scale(canvas_frame, from_=0, to=255, orient="horizontal")
    binary_thresh_slider.set(binary_thresh_value)
    binary_thresh_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Edge Lower Value:").pack(pady=5)
    edge_lower_slider = tk.Scale(canvas_frame, from_=0, to=255, orient="horizontal")
    edge_lower_slider.set(edge_lower_value)
    edge_lower_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Edge Higher Value:").pack(pady=5)
    edge_higher_slider = tk.Scale(canvas_frame, from_=0, to=255, orient="horizontal")
    edge_higher_slider.set(edge_higher_value)
    edge_higher_slider.pack(pady=5)

    # Hough Circles Variables
    ttk.Label(canvas_frame, text="DP Value:").pack(pady=5)
    dp_slider = tk.Scale(canvas_frame, from_=0.1, to=2, orient="horizontal", resolution=0.1)
    dp_slider.set(dp_value)
    dp_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Min Distance:").pack(pady=5)
    min_dist_slider = tk.Scale(canvas_frame, from_=1, to=100, orient="horizontal")
    min_dist_slider.set(min_dist_value)
    min_dist_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Param1 Value:").pack(pady=5)
    param1_slider = tk.Scale(canvas_frame, from_=1, to=200, orient="horizontal")
    param1_slider.set(param1)
    param1_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Param2 Value:").pack(pady=5)
    param2_slider = tk.Scale(canvas_frame, from_=1, to=200, orient="horizontal")
    param2_slider.set(param2)
    param2_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Min Radius:").pack(pady=5)
    min_radius_slider = tk.Scale(canvas_frame, from_=1, to=100, orient="horizontal")
    min_radius_slider.set(min_radius)
    min_radius_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Max Radius:").pack(pady=5)
    max_radius_slider = tk.Scale(canvas_frame, from_=1, to=100, orient="horizontal")
    max_radius_slider.set(max_radius)
    max_radius_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Camera to Target Distance:").pack(pady=5)
    camera_distance_slider = tk.Scale(canvas_frame, from_=0.1, to=100, orient="horizontal", resolution=0.1)
    camera_distance_slider.set(camera_to_target_distance)
    camera_distance_slider.pack(pady=5)

    # Apply Button
    apply_button = ttk.Button(canvas_frame, text="Apply Variables", command=update_variables)
    apply_button.pack(pady=20)

    # Label to show updated values
    result_label = ttk.Label(canvas_frame, text=f"Current Values:\nBlur: {blur_value}\nAdaptive Threshold: {adaptive_thresh_value}\nBinary Threshold: {binary_thresh_value}\n"
                                               f"Edge Lower: {edge_lower_value}\nEdge Higher: {edge_higher_value}\n\n"
                                               f"DP: {dp_value}\nMin Dist: {min_dist_value}\nParam1: {param1}\nParam2: {param2}\n"
                                               f"Min Radius: {min_radius}\nMax Radius: {max_radius}\nCamera Distance: {camera_to_target_distance}")
    result_label.pack(pady=10)

    # Update scroll region whenever the content changes
    canvas_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    top.geometry("350x600")

# Function to detect the USB cameras in a separate thread
def detect_cameras_thread():
    """global camera_detected
    max_cameras = 30  # Assuming we want to check for 5 possible camera indices (0-4)

    while True:
        for camera_id in range(1, max_cameras):
            if (camera_id in camera_detected):
                print("camera added")
                continue
            else:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    camera_detected[camera_id] = True
                    check_camera_and_open_editor(camera_id)
                    cap.release()  # Close the camera after detection
                else:
                    if camera_id in camera_detected:
                        print("camera exist")
                        #del camera_detected[camera_id]  # Remove if the camera was previously detected and is no longer available
        time.sleep(1)"""
    print("called")

# Function to create and store the Camera object based on user input
def create_camera_object(camera_id, lane, target):
    top
    # Validate the input for target and lane
    if not target or not lane:
        messagebox.showerror("Lỗi", f"Hãy nhập đủ các thông tin tên bia, dải bắn của camera {camera_id}")
        return

    # Create a Camera object with the values entered
    camera_obj = camera.Camera(lane, target, camera_id)
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
    top.title(f"Nhập thông tin camera {camera_id}")

    # Create the option menu with the predefined values
    bia_options = ["BiaSo4", "BiaSo8", "BiaSo10"]
    bia_menu = tk.OptionMenu(top, bia_var, *bia_options)
    bia_menu.pack(pady=5)
    ttk.Label(top, text="Lane:").pack(pady=5)
    lane_entry = ttk.Entry(top)
    lane_entry.insert(0, "Enter Lane")  # Default value
    lane_entry.pack(pady=5)
    
    # Create the "Apply and Close" button
    apply_button = ttk.Button(top, text="OK", command=lambda: create_camera_object(camera_id, lane_entry.get(), bia_var.get()))
    apply_button.pack(pady=20)

    top.geometry("300x200")

# Function to check if the camera is detected and update the GUI accordingly
def check_camera_and_open_editor(max_camera):
    for camera_id in range(1, max_camera):
            if (camera_id in camera_detected):
                print("camera added")
            else:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    camera_detected[camera_id] = True
                    cap.release()  # Close the camera after detection
                else:
                    detection_label.config(text=f"Đã nhập {len(camera_detected)} camera")
            if camera_id in camera_detected:
                open_variable_editor(camera_id)
                cv2.waitKey(0)
    
# Label to show camera detection status
detection_label = ttk.Label(root, text=f"Tổng cộng {len(camera_detected)} camera đã thêm")
detection_label.pack(pady=20)

# Start the camera detection in a separate thread
#threading.Thread(target=detect_cameras_thread, daemon=True).start()

def show_result():
    photo1 = photo2 = photo3 = None
    global num_lane, num_turn

    for turn in range(num_turn):
        for lane in range(num_lane):
            result_dir = f"./HinhAnh/KetQua/DaiBan{lane+1}"
            
            if os.path.exists(result_dir):
                try:
                    # Load images using Pillow
                    image1 = Image.open(f"./HinhAnh/KetQua/DaiBan{lane+1}/BiaSo4-{lane+1}-{turn+1}-marked.jpg")
                    image2 = Image.open(f"./HinhAnh/KetQua/DaiBan{lane+1}/BiaSo10-{lane+1}-{turn+1}-marked.jpg")
                    image3 = Image.open(f"./HinhAnh/KetQua/DaiBan{lane+1}/BiaSo8-{lane+1}-{turn+1}-marked.jpg")

                    # Convert the images to a format Tkinter can use
                    photo1 = ImageTk.PhotoImage(image1)
                    photo2 = ImageTk.PhotoImage(image2)
                    photo3 = ImageTk.PhotoImage(image3)

                    new_window = tk.Toplevel(root)
                    new_window.title(f"Kết quả bắn loạt {turn+1}")
                    new_window.geometry("1920x1080")
                    # Create labels and add them to the window
                    label1 = tk.Label(new_window, image=photo1)
                    label1.grid(row=0, column=0)

                    label2 = tk.Label(new_window, image=photo2)
                    label2.grid(row=0, column=1)

                    label3 = tk.Label(new_window, image=photo3)
                    label3.grid(row=0, column=2)
                    label1.image = photo1
                    label2.image = photo2
                    label3.image = photo3
                except Exception as e:
                    print(f"Error loading images: {e}")
                    continue  # Skip to the next iteration if there's an error loading imag

def start_shooting():
    if not num_lane > 0:
        messagebox.showerror("Thông báo", "Hãy thêm dải bắn")
        return
    """
    Begin the shooting process by capturing an image for the safety target.
    """
    # for each lane create a subfolder for containging images
    for lane in range(num_lane):
        lane_dir = f"./HinhAnh/DaiBan{lane+1}"
        result_dir = f"./HinhAnh/KetQua/DaiBan{lane+1}"
        if not os.path.exists(lane_dir):
            os.makedirs(lane_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    # chụp tất cả các bia trước khi bắn để so sánh
    # call parallel capture
    camera.parallel_capture(camera_objects, 0)
    messagebox.showinfo("Thông báo", f"Bắt đầu bắn loạt {num_turn}")

def add_shooting_lane():
    """
    Add a new shooting lane tab to the notebook.
    """
    global num_lane
    num_lane += 1
    messagebox.showinfo("Thông báo", f"Đã thêm 1 dải bắn, tổng cộng {num_lane} dải bắn")
    
    shooting_lane = ttk.Frame(notebook)
    notebook.add(shooting_lane, text=f"Dải bắn {num_lane}")
    label = tk.Label(shooting_lane, text=f"Dải bắn {num_lane}")
    label.pack(pady=20)

def add_shooting_turn():
    """
    Add a new shooting turn.
    """
    global num_turn
    num_turn += 1
    
    messagebox.showinfo("Thông báo", f"Đã thêm 1 Loạt bắn, tổng cộng {num_turn} loạt bắn")

def review_result(img, lane, turn, target):
    print("review result")

def shooting_turn_complete():
    """
    Complete the current shooting turn and capture an image.
    """
    global num_turn, num_lane
    camera.parallel_capture(camera_objects, num_turn)
    cv2.waitKey(1000)
    for lane in range (num_lane):
        for target in targets:
            image_processing.compare_and_detect(lane+1, num_turn, target)

def reset():
    """
    Reset all lanes and turns to their initial state.
    """
    global num_lane, num_turn
    num_lane = 0
    num_turn = 0
    for tab in notebook.tabs():
        notebook.forget(tab)
    messagebox.showinfo("Thông báo", "Reset xong")

def get_current_tab():
    current_tab_id = notebook.select()
    return notebook.nametowidget(current_tab_id)
    
# Create and pack buttons
start_shooting_btn = tk.Button(root, text="Bắt đầu bắn", command=start_shooting)
start_shooting_btn.pack(padx=10, side="left")

add_shooting_lane_btn = tk.Button(root, text="Thêm Dải Bắn", command=add_shooting_lane)
add_shooting_lane_btn.pack(padx=10, side="left")

add_shooting_turn_btn = tk.Button(root, text="Bắt đầu bắn loạt tiếp theo", command=add_shooting_turn)
add_shooting_turn_btn.pack(padx=10, side="left")

shooting_turn_complete_btn = tk.Button(root, text="Báo bia", command=shooting_turn_complete)
shooting_turn_complete_btn.pack(padx=10, side="left")

edit_variables_btn = tk.Button(root, text="Chỉnh sửa tham số", command=open_variable_editor)
edit_variables_btn.pack(padx=10, side="left")

add_camera_btn = tk.Button(root, text="Thêm Camera", command=lambda: check_camera_and_open_editor(10))# cho phép nhập max camera
add_camera_btn.pack(padx=10, side="left")


root.mainloop()
