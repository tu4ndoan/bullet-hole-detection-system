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
max_camera = 10

# Dictionary to store detected cameras
camera_detected = []
camera_objects = []
targets = [] #lets make the user input this


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

# Function to check if the camera is detected and update the GUI accordingly
def check_camera_and_open_editor(max_camera):
    for camera_id in range(1, max_camera):
            if (camera_id in camera_detected):
                print("camera added")
            else:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    camera_detected.append(camera_id)
                    cap.release()  # Close the camera after detection
                    open_variable_editor(camera_id)
                    cv2.waitKey(0)
                else:
                    detection_label.config(text=f"Đã nhập {len(camera_detected)} camera")                
    

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
    print("view all cam")

def show_full_image(lane, turn, target):
    image1 = image_processing.load_result(lane, turn, target)
    cv2.imshow("img", image1)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

def show_result(turn):
    for target in targets: # moi target hien 1 window, show tat ca cac lane
        window = tk.Toplevel(root)
        window.title(f"Ket qua loat {turn}")
        window.geometry("800x600")
        canvas = tk.Canvas(window)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(window, orient="horizontal", command=canvas.xview)
        scrollbar.pack(side="bottom", fill="x")
        canvas.configure(xscrollcommand=scrollbar.set)

        frame = tk.Frame(canvas)
        canvas.create_window((0,0), window=frame, anchor="nw")
        images = []
        for lane in range(1, num_lane+1):
            result_image, result_text = image_processing.compare_and_detect(lane, turn, target)
            images.append((result_image, result_text))
        for result_image, result_text in images:
            #result_image_resized = result_image.resize((100, 100))
            result_image_resized = cv2.resize(result_image, (200,200))
            # Convert the resized NumPy array to a PIL Image
            result_image_pil = Image.fromarray(cv2.cvtColor(result_image_resized, cv2.COLOR_BGR2RGB))

            # Now you can use the PIL Image object with ImageTk.PhotoImage
            img = ImageTk.PhotoImage(result_image_pil)
            # Create a frame for each image and its description
            img_desc_frame = tk.Frame(frame)
            img_desc_frame.pack(side="left", padx=10, pady=10)

            # Create a label for the image
            img_label = tk.Label(img_desc_frame, image=img)
            img_label.image = img  # Keep a reference to the image to prevent garbage collection
            img_label.pack()

            # Create a label for the description text
            desc_label = tk.Label(img_desc_frame, text=result_text)
            desc_label.pack()

            # Bind the image label to open the full image on click
            img_label.bind("<Button-1>", lambda event, lane=lane, turn=turn, target=target: show_full_image(lane, turn, target))

        # Update the scrollable region of the canvas
        canvas.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
    
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

def remove_shooting_lane(lane_num):
    print("")

def add_shooting_turn():
    """
    Add a new shooting turn.
    """
    global num_turn
    num_turn += 1
    
    messagebox.showinfo("Thông báo", f"Đã thêm 1 Loạt bắn, tổng cộng {num_turn} loạt bắn")

def shooting_turn_complete():
    """
    Complete the current shooting turn and capture an image.
    """
    global num_turn, num_lane
    # capture the target after each shooting turn
    camera.parallel_capture(camera_objects, num_turn)
    cv2.waitKey(1000)
    # bao bia
    show_result(num_turn)
    for lane in range(num_lane):
        print("called bao bia")
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

#edit_variables_btn = tk.Button(root, text="Mở thư mục Kết Quả", command=open_variable_editor)
#edit_variables_btn.pack(padx=10, side="left")

add_camera_btn = tk.Button(root, text="Thêm Camera", command=lambda: check_camera_and_open_editor(max_camera))# cho phép nhập max camera
add_camera_btn.pack(padx=10, side="left")

check_camera_btn = tk.Button(root, text="Kiểm tra camera", command=view_all_camera)
check_camera_btn.pack(padx=10, side="left")

detection_label = ttk.Label(root, text=f"Tổng cộng {len(camera_detected)} camera đã thêm")
detection_label.pack(pady=20)

root.mainloop()
