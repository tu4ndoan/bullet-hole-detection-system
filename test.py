images = [
        ("./HinhAnh/KetQua/DaiBan1/BiaSo4-1-1-marked.jpg", "This is image 1"),
        ("./HinhAnh/KetQua/DaiBan1/BiaSo4-1-2-marked.jpg", "This is image 2"),
        ("./HinhAnh/KetQua/DaiBan1/BiaSo4-1-3-marked.jpg", "This is image 3"),
    ]
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def show_full_image(img_path):
    # Create a new window to display the full-size image
    full_size_window = tk.Toplevel()  # Toplevel creates a new window
    full_size_window.title("Full Size Image")

    # Open the image using PIL
    pil_image = Image.open(img_path)
    img = ImageTk.PhotoImage(pil_image)

    # Create a label to display the image in the new window
    img_label = tk.Label(full_size_window, image=img)
    img_label.image = img  # Keep a reference to the image to prevent garbage collection
    img_label.pack(padx=10, pady=10)

    # Run the event loop for the new window
    full_size_window.mainloop()

def create_image_window():
    # Create a new Tkinter window
    window = tk.Tk()
    window.title("Image Gallery with Descriptions")

    # Create a Canvas widget to hold the images and a horizontal scrollbar
    canvas = tk.Canvas(window)
    canvas.pack(side="left", fill="both", expand=True)

    # Create a Scrollbar widget and link it to the Canvas
    scrollbar = ttk.Scrollbar(window, orient="horizontal", command=canvas.xview)
    scrollbar.pack(side="bottom", fill="x")
    canvas.configure(xscrollcommand=scrollbar.set)

    # Create a Frame inside the Canvas to hold the images and descriptions
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    # List of image file paths and corresponding descriptions
    images = [
        ("./HinhAnh/KetQua/DaiBan1/BiaSo4-1-1-marked.jpg", "Bệ số 1:\n9-8-0 \n17\nKhá"),
        ("./HinhAnh/KetQua/DaiBan1/BiaSo4-1-2-marked.jpg", "Bệ số 1:\n9-8-0 \n17\nKhá"),
        ("./HinhAnh/KetQua/DaiBan1/BiaSo4-1-3-marked.jpg", "Bệ số 1:\n9-8-0 \n17\nKhá"),
        ("./HinhAnh/KetQua/DaiBan1/BiaSo4-1-1-marked.jpg", "Bệ số 1:\n9-8-0 \n17\nKhá"),
        ("./HinhAnh/KetQua/DaiBan1/BiaSo4-1-2-marked.jpg", "Bệ số 1:\n9-8-0 \n17\nKhá"),
        ("./HinhAnh/KetQua/DaiBan1/BiaSo4-1-3-marked.jpg", "Bệ số 1:\n9-8-0 \n17\nKhá"),
        ("./HinhAnh/KetQua/DaiBan1/BiaSo4-1-1-marked.jpg", "Bệ số 1:\n9-8-0 \n17\nKhá"),
        ("./HinhAnh/KetQua/DaiBan1/BiaSo4-1-2-marked.jpg", "Bệ số 1:\n9-8-0 \n17\nKhá"),
        ("./HinhAnh/KetQua/DaiBan1/BiaSo4-1-3-marked.jpg", "Bệ số 1:\n9-8-0 \n17\nKhá"),
    ]

    # Loop through images and create image widgets and description labels
    for img_path, description in images:
        # Open and convert the image using PIL
        pil_image = Image.open(img_path)
        pil_image_resized = pil_image.resize((100, 100))  # Resize the image if necessary
        img = ImageTk.PhotoImage(pil_image_resized)

        # Create a frame for each image and its description
        img_desc_frame = tk.Frame(frame)
        img_desc_frame.pack(side="left", padx=10, pady=10)

        # Create a label for the image
        img_label = tk.Label(img_desc_frame, image=img)
        img_label.image = img  # Keep a reference to the image to prevent garbage collection
        img_label.pack()

        # Create a label for the description text
        desc_label = tk.Label(img_desc_frame, text=description)
        desc_label.pack()

        # Bind the image label to open the full image on click
        img_label.bind("<Button-1>", lambda event, path=img_path: show_full_image(path))

    # Update the scrollable region of the canvas
    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    # Run the Tkinter event loop
    window.mainloop()

# Call the function to create and show the window
create_image_window()





#img1 = cam_1.capture_image(15)
#cv2.waitKey(5000)
#img2 = cam_1.capture_image(21)
#image_processing.compare_and_detect(1, 21, "BiaSo4")
# TODO: handle case 2 lo dan gan nhau hoac de len nhau
# => lỗ đạn sẽ to hơn lỗ đạn bình thường
# => nếu cùng 1 loạt thì so contour area
# => nêu khác loạt thì phát hiện được đơn giản
# TODO: handle case 2 lo dan trung nhau
# => diff thresh sẽ cực nhỏ hoặc = 0

# TODO: báo điểm tất cả dải bắn trong loạt (sửa scrollbar và implement vào main app)
# TODO: tính điểm (co sai lech +- 1 diem) xong
#image_processing.get_center_ellipse_parameters(img2)
# TODO: cho chon tam bia bang tay
# hoac xac dinh ellipse bang fit ellipse nhu truoc
# xong cho dich tam bia xuong theo moi vong tu i>=3 k+=5*i