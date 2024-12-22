import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import camera_connect
import os
import camera
import add_text
import testcamera
import image_processing

# Local Variable
program_name = "Tiểu đoàn 1038 - Báo bia bằng camera"
# Create main window
root = tk.Tk()
root.title(program_name)
root.geometry("800x600")

# Create a Notebook widget to hold tabs
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)
num_lane = len(notebook.tabs())
num_turn = 1
label = ttk.Frame(notebook)

targets = ["BiaSo4", "BiaSo10", "BiaSo8"] #lets make the user input this

def show_result():
    photo1 = photo2 = photo3 = None
    global num_lane, num_turn

    for turn in range(num_turn):
        for lane in range(num_lane):
            result_dir = f"./Images/Result/Lane{lane+1}"
            
            if os.path.exists(result_dir):
                try:
                    # Load images using Pillow
                    image1 = Image.open(f"./Images/Result/Lane{lane+1}/BiaSo4-{lane+1}-{turn+1}-marked.jpg")
                    image2 = Image.open(f"./Images/Result/Lane{lane+1}/BiaSo10-{lane+1}-{turn+1}-marked.jpg")
                    image3 = Image.open(f"./Images/Result/Lane{lane+1}/BiaSo8-{lane+1}-{turn+1}-marked.jpg")

                    # Convert the images to a format Tkinter can use
                    photo1 = ImageTk.PhotoImage(image1)
                    photo2 = ImageTk.PhotoImage(image2)
                    photo3 = ImageTk.PhotoImage(image3)

                    new_window = tk.Toplevel(root)
                    new_window.title(f"Ket Qua Ban Loat {turn+1}, Be So {lane+1}")
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
        lane_dir = f"./Images/Lane{lane+1}"
        result_dir = f"./Images/Result/Lane{lane+1}"
        if not os.path.exists(lane_dir):
            os.makedirs(lane_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    # chụp tất cả các bia trước khi bắn để so sánh
    camera_connect.parallel_capture()
    messagebox.showinfo("Thông báo", "Bắt đầu bắn")


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
    num_turn = num_turn + 1
    messagebox.showinfo("Thông báo", f"Đã thêm 1 Loạt bắn, tổng cộng {num_turn} loạt bắn")

def review_result(img, lane, turn, target):
    print("review result")

def process_and_save_result(lane, turn, target):
    #img = image_processing.load_image(lane, turn, target)
    test_img = cv2.imread("./Images/Lane1/test.jpg")# just for testing
    
    image_processing.detect_bullet_hole(test_img, turn, lane)


def shooting_turn_complete():
    """
    Complete the current shooting turn and capture an image.
    """
    global num_turn, num_lane
    for lane in range(num_lane):
        print(f"loat thu {num_turn}, dai ban {lane+1}")
        for target in targets:
            process_and_save_result(lane+1, num_turn, target)
                
    #show_result()
    messagebox.showinfo("Thông báo", "Xem kết quả bắn tại thư mục Result")

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

def capture_images():
    camera_connect.parallel_capture()
    messagebox.showinfo("Thông báo", "Đã chụp/lưu ảnh tại thư mục Images/")

def on_submit(param1, param2, param3, window):
    if (param1, param2, param3, window):
        print(f"Parameters received: {param1}, {param2}, {param3}")
        add_camera(param1, param2, param3)

    window.destroy()

def display_camera(camera_id):
    newwin = tk.Toplevel(root)
    newwin.title("new window")
    newwin.geometry("800x600")
    cam_indices = [0]
    cam = camera.CameraApp(newwin, newwin, cam_indices)
    
def add_camera(lane, target, camera_id):
    testcamera.add_camera(lane, target, camera_id, get_current_tab(), root)

def add_camera_form():
    new_window = tk.Toplevel(root)
    new_window.title("Them camera")
    new_window.geometry("800x600")
    label1 = tk.Label(new_window, text="Dải bắn sô:")
    label1.pack(pady=5)
    shooting_lane = tk.Entry(new_window)
    shooting_lane.pack(pady=5)

    label2 = tk.Label(new_window, text="Mục tiêu:")
    label2.pack(pady=5)
    target_name = tk.Entry(new_window)
    target_name.pack(pady=5)

    label3 = tk.Label(new_window, text="Camera ID:")
    label3.pack(pady=5)
    camera_id = tk.Entry(new_window)
    camera_id.pack(pady=5)


    submit_button = tk.Button(new_window, text="Thêm", command=lambda: on_submit(shooting_lane.get(), target_name.get(), camera_id.get(), new_window))
    submit_button.pack(pady=20)

    
# Create and pack buttons
start_shooting_btn = tk.Button(root, text="Bắt đầu bắn", command=start_shooting)
start_shooting_btn.pack(padx=10, side="left")

add_shooting_lane_btn = tk.Button(root, text="Thêm Dải Bắn", command=add_shooting_lane)
add_shooting_lane_btn.pack(padx=10, side="left")

add_shooting_turn_btn = tk.Button(root, text="Bắt đầu bắn loạt tiếp theo", command=add_shooting_turn)
add_shooting_turn_btn.pack(padx=10, side="left")

shooting_turn_complete_btn = tk.Button(root, text="Bắn xong 1 loạt", command=shooting_turn_complete)
shooting_turn_complete_btn.pack(padx=10, side="left")

reset_btn = tk.Button(root, text="Xem kết quả bắn", command=show_result)
reset_btn.pack(padx=10, side="left")

capture_images_btn = tk.Button(root, text="Chụp & Lưu ảnh", command=capture_images)
capture_images_btn.pack(padx=10, side="left")

add_camera_btn = tk.Button(root, text="Add Camera", command=add_camera_form)
add_camera_btn.pack(padx=10, side="left")


root.mainloop()
