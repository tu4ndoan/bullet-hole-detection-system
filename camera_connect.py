import cv2
from PIL import Image, ImageTk
import tkinter as tk
import os
import threading

def capture_image(cam_index):
    cap = cv2.VideoCapture(cam_index)
    
    output_dir = './Images/Lane{lane_num}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not cap.isOpened():
        print(f"Error: Camera {cam_index} not found.")
        return

    ret, frame = cap.read()
    if ret:
        image_path = os.path.join(output_dir, f"camera_{cam_index}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Captured image from Camera {cam_index}: {image_path}")
    else:
        print(f"Error: Could not capture image from Camera {cam_index}")
    
    cap.release()

def parallel_capture():
    # List to store threads
    threads = []

    # Create threads for each camera
    for cam_index in range(2):
        t = threading.Thread(target=capture_image, args=(cam_index,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    cv2.destroyAllWindows()