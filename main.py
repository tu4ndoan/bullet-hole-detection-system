import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import os
import camera
import image_processing
import cv2
import detect_bullet_hole

# https://viblo.asia/p/u-net-kien-truc-manh-me-cho-segmentation-1Je5Em905nL
# Create main window
root = tk.Tk()
root.title("TIỂU ĐOÀN 1038 - ỨNG DỤNG BÁO BIA TỰ ĐỘNG")
root.geometry("800x600")

# Create a Notebook widget to hold tabs, each shooting lane is 1 tab
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)
label = ttk.Frame(notebook)
lane_canvases = {}

# Variables
num_turn = 1
max_camera = 100
# Boolean variable to store the state of the checkbox
b_debug = tk.BooleanVar()
b_debug.set(False)  # Set the default state of the checkbox

# Dictionary to store detected cameras

def get_num_lane():
    return len(notebook.tabs())

# Function to check if the camera is detected and update the GUI accordingly
def check_camera_and_open_editor(max_camera):
    for camera_id in range(1, max_camera):
        
        if (camera_id in camera.cameras):
            print(f"Checking camera {camera.cameras[camera_id].camera_id} - {camera.cameras[camera_id].target} - {camera.cameras[camera_id].lane}")
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print(f"Error: Camera {camera_id} đã mất kết nối, hãy kiểm tra lại")
                camera.cameras[camera_id].deactivate()
                print(camera.cameras)

            else:
                print(f"Camera {camera_id} check OK")
                cap.release()
        else:
            try:
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                if cap.isOpened():
                    print(f"camera {camera_id}: cap is opened")
                    print("Opening camera editor")
                    camera.open_variable_editor(camera_id)
                    cap.release()
                    cv2.waitKey(0)
                else:
                    print(f"Đã nhận diện hết camera, tổng cộng {len(camera.cameras)} camera")
                    detection_label.config(text=f"Đã nhập {len(camera.cameras)} camera")
                    break
            except Exception as e:
                print(f"Error with camera index {camera_id}: {e}")
                break
                            
    
def show_full_image(img, lane, turn):
    #image1 = image_processing.load_result(lane, turn, target)
    cv2.imshow(f"Loat {turn}, Be so {lane}", img)
    cv2.waitKey(0)

def get_result_for_lane(lane, turn):
    result = []
    for target in camera.targets:
        try:
            result_image, result_text, total_score = detect_bullet_hole.compare_and_detect(lane, turn, target)

            result.append((result_image, result_text, total_score))
            print(f"loạt {turn} bệ số {lane}: {result_text}-{total_score}")
        except Exception as e:
            print(e)
    return result

def add_result_to_frame(lane, turn):
    new_result = get_result_for_lane(lane, turn)
    canvas, content_frame, current_column = get_canvas_by_lane(lane)
    # Add the new images to the content_frame
    current_column = 1
    side_frame = tk.Frame(content_frame)
    side_label = tk.Label(side_frame, text=f"Loạt {turn}")
    side_label.pack()
    side_frame.grid(row=turn-1, column=0, padx=10, pady=10)
    for image, text, _ in new_result:
        result_image_resized = cv2.resize(image, (200, 200))
        
        result_image_pil = Image.fromarray(cv2.cvtColor(result_image_resized, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(result_image_pil)
        
        frame = tk.Frame(content_frame)
        img_label = tk.Label(frame, image=img)
        img_label.image = img  # Keep a reference to the image
        img_label.pack()
        # Bind the image click event to the on_image_click function
        img_label.bind("<Button-1>", lambda event, lane=lane, turn=turn, img=image, text=text: show_full_image(img, lane, turn))
        
        desc_label = tk.Label(frame, text=text)
        desc_label.pack()
        frame.grid(row=turn-1, column=current_column, padx=10, pady=10)
        
        current_column += 1
    lane_canvases[lane] = (canvas, content_frame, current_column)
        # Update the scrollable region of the canvas to include the new images
    content_frame.update_idletasks()  # Update content frame size
    canvas.config(scrollregion=canvas.bbox("all"))  # Update the scroll region
    
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
            print(f"Creating directory {lane_dir}")
            os.makedirs(lane_dir)
        if not os.path.exists(result_dir):
            print(f"Creating directory {result_dir}")
            os.makedirs(result_dir)
    # chụp tất cả các bia trước khi bắn để so sánh
    # call parallel capture
    if not b_debug.get():
        print("Bắt đầu bắn loạt 1, đang chụp ảnh toàn bộ bia trước khi bắn, các loạt bắn sau lưu ý không bấm lại nút Bắt đầu bắn")
        camera.parallel_capture(0)

    messagebox.showinfo("Thông báo", f"Bắt đầu bắn loạt {num_turn}")

def get_canvas_by_lane(lane):
    """
    Get the canvas for the given lane.
    """
    return lane_canvases.get(lane)

def add_shooting_lane():
    """
    Add a new shooting lane tab to the notebook.
    """
    lane_number = get_num_lane() + 1  # New lane number
    
    # Create a new tab for the new lane
    shooting_lane = ttk.Frame(notebook)
    notebook.add(shooting_lane, text=f"Dải bắn {lane_number}")
    
    label = tk.Label(shooting_lane, text=f"Dải bắn {lane_number}")
    label.pack(pady=5)

    remove_btn = tk.Button(shooting_lane, text="Xóa dải bắn", command=lambda: remove_shooting_lane())
    remove_btn.pack(pady=5)

    # Create the canvas that will hold all the images
    canvas = tk.Canvas(shooting_lane)
    canvas.pack(side="left", fill="both", expand=True)

    # Create a horizontal scrollbar for the canvas
    scrollbar = ttk.Scrollbar(canvas, orient="horizontal", command=canvas.xview)
    scrollbar.pack(side="bottom", fill="x")
    v_scrollbar = ttk.Scrollbar(canvas, orient="vertical", command=canvas.yview)
    v_scrollbar.pack(side="right", fill="y")
    canvas.configure(xscrollcommand=scrollbar.set)
    canvas.configure(yscrollcommand=v_scrollbar.set)
    # Create a frame inside the canvas to hold the images
    content_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=content_frame, anchor="nw")

    # Store the canvas in the dictionary with lane number as the key
    lane_canvases[lane_number] = (canvas, content_frame, 0)


def remove_shooting_lane():
    lane = get_current_tab()
    notebook.forget(lane)

def add_shooting_turn():
    """
    Add a new shooting turn.
    """
    global num_turn
    num_turn += 1
    turn_label.config(text=f"Loạt {num_turn}")
    messagebox.showinfo("Thông báo", f"Bắt đầu bắn loạt {num_turn}")

def shooting_turn_complete():
    """
    Complete the current shooting turn and capture an image.
    """
    global num_turn
    # capture the target after each shooting turn
    if not b_debug.get():
        print(f"Đang chụp ảnh báo bia loạt {num_turn}...")
        camera.parallel_capture(num_turn)
        cv2.waitKey(5000)
        print(f"Đã chụp xong ảnh báo bia loạt {num_turn}")
    # bao bia
    for lane in range(1, get_num_lane()+1):
        print(f"Đang xử lý kết quả loạt {num_turn} bệ số {lane}...")
        add_result_to_frame(lane, num_turn)
        print(f"Đã xử lý xong kết quả loạt {num_turn} bệ số {lane}")

def reset():
    """
    Reset all lanes and turns to their initial state.
    """
    print("Đang reset loạt bắn và dải bắn, camera giữ nguyên")
    global num_turn
    num_turn = 0
    turn_label.config(text=f"Loạt {num_turn}")
    for tab in notebook.tabs():
        notebook.forget(tab)
    print("Reset complete")
    messagebox.showinfo("Thông báo", "Reset xong")

def get_current_tab():
    current_tab_id = notebook.select()
    return notebook.nametowidget(current_tab_id)
    
def open_result_folder():
    print(os.path.abspath("./HinhAnh/KetQua/"))
    os.startfile(os.path.abspath("./HinhAnh/KetQua/"))

def view_all_camera():
    camera.view_all_camera()

def view_result(turn):
    # create a new window to hold result for all lane
    # Create the canvas that will hold all the images
    res_win = tk.Toplevel(root)
    res_can = tk.Canvas(res_win)
    title = f"Kết quả loạt {turn}"
    res_win.title(title)
    res_win.geometry("800x600")
    res_can.pack(side="left", fill="both", expand=True)

    # Create a horizontal scrollbar for the canvas
    scrollbar = ttk.Scrollbar(res_can, orient="horizontal", command=res_can.xview)
    scrollbar.pack(side="bottom", fill="x")
    v_scrollbar = ttk.Scrollbar(res_can, orient="vertical", command=res_can.yview)
    v_scrollbar.pack(side="right", fill="y")
    res_can.configure(xscrollcommand=scrollbar.set)
    res_can.configure(yscrollcommand=v_scrollbar.set)
    # Create a frame inside the canvas to hold the images
    content_frame = tk.Frame(res_can)
    res_can.create_window((0, 0), window=content_frame, anchor="nw")
    for lane in range(1, get_num_lane()+1):
        top_frame = tk.Frame(content_frame)
        label = tk.Label(top_frame, text=f"Bệ số {lane}")
        label.pack()
        top_frame.grid(row=0, column=lane-1, padx=10, pady=10)
        current_row = 1
        # each lane, get the result and display
        total_score = 0
        
        for target in camera.targets: 
            # each lane has N targets, load N images, grid row = 0, 1, 2
            # row 3 is total score of all targets of this lane, turn
            target_result = None
            for result in detect_bullet_hole.results:
                if result["name"] == f"{lane}-{turn}-{target}":
                    target_result = result
            total_score += target_result["total_score"] # total score (all targets) of this lane
            text = target_result["result_text"]
            image = image_processing.load_result(lane, turn, target)
            result_image_resized = cv2.resize(image, (200, 200))
            
            result_image_pil = Image.fromarray(cv2.cvtColor(result_image_resized, cv2.COLOR_BGR2RGB))
            img = ImageTk.PhotoImage(result_image_pil)
            
            frame = tk.Frame(content_frame)
            img_label = tk.Label(frame, image=img)
            img_label.image = img  # Keep a reference to the image
            img_label.pack()
            
            label = tk.Label(frame, text=text)
            label.pack()
            # Bind the image click event to the on_image_click function
            img_label.bind("<Button-1>", lambda event, lane=lane, turn=turn, img=image: show_full_image(img, lane, turn))
            frame.grid(row=current_row, column=lane-1, padx=10, pady=10)
            current_row += 1
        # Update the scrollable region of the canvas to include the new images
        total_score_frame = tk.Frame(content_frame)
        if total_score < 45:
            grade = "Không đạt"
        elif 45 <= total_score <= 58:
            grade = "Đạt"
        elif 59 <= total_score <= 71:
            grade = "Khá"
        elif 72 <= total_score <= 90:
            grade = "Giỏi"
        else:
            grade = ""
        score_label = tk.Label(total_score_frame, text=f"Tổng: {total_score} điểm - {grade}")
        score_label.pack()
        total_score_frame.grid(row=current_row, column=lane-1, padx=10, pady=10)
    content_frame.update_idletasks()  # Update content frame size
    res_can.config(scrollregion=res_can.bbox("all"))  # Update the scroll region

def toggle_debug():
    if b_debug.get():
        print("Bạn đang kích hoạt Chế độ debug để xem các phân tích hình ảnh chuyên môn cao của computer vision, nếu bạn không phải chuyên gia hãy tắt chế độ này.")
    else:
        print("Chế độ debug đã tắt.")
    detect_bullet_hole.b_debug = b_debug.get()

# Create and pack buttons
start_shooting_btn = tk.Button(root, text="Bắt đầu bắn", command=start_shooting)
start_shooting_btn.pack(padx=5, side="left")

add_shooting_lane_btn = tk.Button(root, text="Thêm Dải Bắn", command=add_shooting_lane)
add_shooting_lane_btn.pack(padx=5, side="left")

add_shooting_turn_btn = tk.Button(root, text="Bắt đầu bắn loạt tiếp theo", command=add_shooting_turn)
add_shooting_turn_btn.pack(padx=5, side="left")

shooting_turn_complete_btn = tk.Button(root, text="Báo bia", command=shooting_turn_complete)
shooting_turn_complete_btn.pack(padx=5, side="left")

edit_variables_btn = tk.Button(root, text="Mở thư mục Kết Quả", command=open_result_folder)
edit_variables_btn.pack(padx=5, side="left")

view_result_btn = tk.Button(root, text=f"Thông báo kết quả loạt", command=lambda: view_result(num_turn))
view_result_btn.pack(padx=5, side="left")

add_camera_btn = tk.Button(root, text="Thêm Camera", command=lambda: check_camera_and_open_editor(max_camera))
add_camera_btn.pack(padx=5, side="left")

check_camera_btn = tk.Button(root, text="Kiểm tra camera", command=view_all_camera)
check_camera_btn.pack(padx=5, side="left")

reset_btn = tk.Button(root, text="Reset", command=reset)
reset_btn.pack(padx=5, side="left")

detection_label = ttk.Label(root, text=f"Tổng cộng {len(camera.cameras)} camera đã thêm")
detection_label.pack(pady=5)

turn_label = ttk.Label(root, text=f"Loạt {num_turn}")
turn_label.pack(pady=5)

#checkbox = tk.Checkbutton(root, text="debug mode", variable=b_debug, command=toggle_debug)
#checkbox.pack(padx=5, side="left")



root.mainloop()
