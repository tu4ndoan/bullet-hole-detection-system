import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

root = tk.Tk()
root.title("test")
root.geometry("800x600")
# create img for testing
img0=cv2.imread("./Images/Lane1/BiaSo4-1-0.jpg")
cv2.imwrite(f"./Images/Lane1/BiaSo4-1-1.jpg", img0)

img1=cv2.imread("./Images/Lane1/BiaSo8-1-0.jpg")
cv2.imwrite(f"./Images/Lane1/BiaSo8-1-1.jpg", img1)

img2=cv2.imread("./Images/Lane1/BiaSo10-1-0.jpg")
cv2.imwrite(f"./Images/Lane1/BiaSo10-1-1.jpg", img2)

def create_test_img1(file_name):
    img = cv2.imread(f"./Images/Lane1/{file_name}-1-1.jpg")
    cv2.imwrite(f"./Images/Lane1/{file_name}-1-2.jpg", img)

def create_test_img2(file_name):
    img = cv2.imread(f"./Images/Lane1/{file_name}-1-2.jpg")
    cv2.imwrite(f"./Images/Lane1/{file_name}-1-3.jpg", img)

def create_image_1():
    create_test_img1("BiaSo8")

def create_image_2():
    create_test_img2("BiaSo8")

root = tk.Tk()

create_btn_1 = tk.Button(root, text="create img turn 1", command=create_image_1)
create_btn_1.pack(padx=10, side="left")

create_btn_2 = tk.Button(root, text="create img turn 2", command=create_image_2)
create_btn_2.pack(padx=10, side="left")
