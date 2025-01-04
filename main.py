import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import os
import camera
import image_processing
import cv2
import detect_bullet_hole

# Create main window
root = tk.Tk()
root.title("TIỂU ĐOÀN 1038 - PHẦN MỀM BÁO BIA TỰ ĐỘNG")
root.geometry("800x600")

# Create a Notebook widget to hold tabs, each shooting lane is 1 tab
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)
label = ttk.Frame(notebook)

# Variables
num_turn = 1
max_camera = 10 # bài 1 mỗi dải bắn nhập 3 camera tương đương với 3 bia


# Dictionary to store detected cameras
camera_detected = []


def get_num_lane():
    return len(notebook.tabs())

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
                    camera.open_variable_editor(camera_id)
                    cv2.waitKey(0)
                else:
                    detection_label.config(text=f"Đã nhập {len(camera_detected)} camera")            
    
def show_full_image(lane, turn, target):
    image1 = image_processing.load_result(lane, turn, target)
    cv2.imshow(f"Loat {turn}, Be so {lane}, Muc tieu {target}", image1)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

def get_notebook_tab(lane):
    notebook_tab = notebook.tabs()[lane-1]
    return notebook.nametowidget(notebook_tab)

def show_result(turn):
    for target in camera.targets:  # Process each target
        for lane in range(1, get_num_lane() + 1):
            # Create a new tab for each target
            target_tab = get_notebook_tab(lane)
            canvas = tk.Canvas(target_tab)
            canvas.pack(side="left", fill="both", expand=True)

            # Create horizontal scrollbar for the canvas
            scrollbar = ttk.Scrollbar(target_tab, orient="horizontal", command=canvas.xview)
            scrollbar.pack(side="bottom", fill="x")
            canvas.configure(xscrollcommand=scrollbar.set)

            frame = tk.Frame(canvas)
            canvas.create_window((0, 0), window=frame, anchor="nw")

            result_image, result_text = detect_bullet_hole.compare_and_detect(lane, turn, target)

            # Resize the image to fit into the canvas
            result_image_resized = cv2.resize(result_image, (200, 200))

            # Convert the NumPy array to a PIL Image
            result_image_pil = Image.fromarray(cv2.cvtColor(result_image_resized, cv2.COLOR_BGR2RGB))

            # Convert the PIL Image to a PhotoImage object
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
    if not get_num_lane() > 0:
        messagebox.showerror("Thông báo", "Hãy thêm dải bắn")
        return
    """
    Begin the shooting process by capturing an image for the safety target.
    """
    # for each lane create a subfolder for containging images
    for lane in range(get_num_lane()):
        lane_dir = f"./HinhAnh/DaiBan{lane+1}"
        result_dir = f"./HinhAnh/KetQua/DaiBan{lane+1}"
        if not os.path.exists(lane_dir):
            os.makedirs(lane_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    # chụp tất cả các bia trước khi bắn để so sánh
    # call parallel capture
    camera.parallel_capture(0)
    messagebox.showinfo("Thông báo", f"Bắt đầu bắn loạt {num_turn}")

def add_shooting_lane():
    """
    Add a new shooting lane tab to the notebook.
    """
    shooting_lane = ttk.Frame(notebook)
    notebook.add(shooting_lane, text=f"Dải bắn {get_num_lane() + 1}")
    
    label = tk.Label(shooting_lane, text=f"Dải bắn {get_num_lane()}")
    label.pack(pady=20)
    messagebox.showinfo("Thông báo", f"Đã thêm 1 dải bắn, tổng cộng {get_num_lane()} dải bắn")

def remove_shooting_lane(lane_num):
    lane = notebook.tabs()[lane_num]
    notebook.forget(lane)

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
    global num_turn
    # capture the target after each shooting turn
    camera.parallel_capture(num_turn)
    cv2.waitKey(1000)
    # bao bia
    show_result(num_turn)

def reset():
    """
    Reset all lanes and turns to their initial state.
    """
    global num_turn
    num_turn = 0
    for tab in notebook.tabs():
        notebook.forget(tab)
    messagebox.showinfo("Thông báo", "Reset xong")

def get_current_tab():
    current_tab_id = notebook.select()
    return notebook.nametowidget(current_tab_id)
    
def open_result_folder():
    os.startfile("./HinhAnh/KetQua")

def view_all_camera():
    camera.view_all_camera()

# Create and pack buttons
start_shooting_btn = tk.Button(root, text="Bắt đầu bắn", command=start_shooting)
start_shooting_btn.pack(padx=10, side="left")

add_shooting_lane_btn = tk.Button(root, text="Thêm Dải Bắn", command=add_shooting_lane)
add_shooting_lane_btn.pack(padx=10, side="left")

add_shooting_turn_btn = tk.Button(root, text="Bắt đầu bắn loạt tiếp theo", command=add_shooting_turn)
add_shooting_turn_btn.pack(padx=10, side="left")

shooting_turn_complete_btn = tk.Button(root, text="Báo bia", command=shooting_turn_complete)
shooting_turn_complete_btn.pack(padx=10, side="left")

edit_variables_btn = tk.Button(root, text="Mở thư mục Kết Quả", command=open_result_folder)
edit_variables_btn.pack(padx=10, side="left")

add_camera_btn = tk.Button(root, text="Thêm Camera", command=lambda: check_camera_and_open_editor(max_camera))# cho phép nhập max camera
add_camera_btn.pack(padx=10, side="left")

check_camera_btn = tk.Button(root, text="Kiểm tra camera", command=view_all_camera)
check_camera_btn.pack(padx=10, side="left")

detection_label = ttk.Label(root, text=f"Tổng cộng {len(camera_detected)} camera đã thêm")
detection_label.pack(pady=20)

root.mainloop()
