import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from tkinter import simpledialog

def load_image(lane, turn, target_name):
    image = cv2.imread(f'./Images/Lane{lane}/{target_name}-{lane}-{turn}.jpg')
    return image

def load_result(lane, turn, target_name):
    result = cv2.imread(f'./Images/Result/Lane{lane}/{target_name}-{lane}-{turn}-marked.jpg')
    return result

def preprocess_images(image_1, image):
    """Convert images to grayscale and compute absolute difference."""
    gray_original = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_modified = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(gray_original, gray_modified)
    diff_normalized = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)
    return diff_normalized

def apply_threshold_and_blur(diff_normalized):
    """Threshold and apply Gaussian blur to highlight changes."""
    scaled_diff = diff_normalized // 2  # Reduce sensitivity
    _, thresholded_diff = cv2.threshold(scaled_diff, 50, 255, cv2.THRESH_BINARY)
    blurred_diff = cv2.GaussianBlur(thresholded_diff, (5, 5), 0)
    return blurred_diff, thresholded_diff

def find_and_draw_contours(image, thresholded_diff, turn):
    """Find contours and draw bounding boxes for significant changes."""
    contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 15 or area > 100:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 10:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
           continue
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < 0.5:
            continue  # Not a valid circle
        valid_contours.append(contour)

    for contour in valid_contours:
        if cv2.contourArea(contour) > 1:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)  # Green rectangle
            draw_number(image, x, y, w, h, turn)
    return image

def draw_number(image, x, y, w, h, turn_num):
    """Draw a number on the image near the detected contour."""
    number = turn_num  # Example number
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)  # Red color for number
    thickness = 2
    posX = x + w // 4
    posY = y + (4 * h) // 5
    cv2.putText(image, str(number), (posX, posY), font, font_scale, color, thickness)

def compare_images(image_1, image_2, lane, turn, target_name):
    """Compare two images, detect changes, and mark bullet holes."""
    diff_normalized = preprocess_images(image_1, image_2)
    blurred_diff, thresholded_diff = apply_threshold_and_blur(diff_normalized)
    processed_image = find_and_draw_contours(blurred_diff, thresholded_diff, turn)
    #save_image(processed_image, lane, turn, target_name)
    cv2.imshow(image_2)
    cv2.imshow(image_1)
    cv2.imshow(processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def is_point_inside_ellipse(x, y, h, k, a, b):
    """Check if a point (x, y) is inside an ellipse with center (h, k),
    semi-major axis a, and semi-minor axis b."""
    
    # Apply the ellipse equation
    equation_result = ((x - h) ** 2) / (a ** 2) + ((y - k) ** 2) / (b ** 2)
    # h,k: tam bia
    # a: canh doc
    # b: canh ngang
    # If the result is less than or equal to 1, the point is inside or on the ellipse
    return equation_result <= 1

def get_bullet_holes(lane, turn):
    for result in results:
        if result["name"] == f"{lane}-{turn}":
            return result["holes"]

def get_circle_target_center(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred, 
                                cv2.HOUGH_GRADIENT, 
                                dp=1, 
                                minDist=50, 
                                param1=50, 
                                param2=30, 
                                minRadius=10, 
                                maxRadius=100)

    # If circles are detected
    if circles is not None:
        # Convert circles to integer values
        circles = np.round(circles[0, :]).astype("int")
        for circle in circles:
            # Extract center coordinates and radius
            center_x, center_y, radius = circle

def get_elipse_target_center(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours and fit ellipses
    for contour in contours:
        if len(contour) >= 5:  # Fit an ellipse requires at least 5 points
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)

            # Extract ellipse parameters (center, axes, angle)
            center, axes, angle = ellipse

            # Compute the aspect ratio (major axis / minor axis) to check if it's "perfect"
            aspect_ratio = max(axes) / min(axes)

            # If the aspect ratio is close to 1, it is more likely a perfect circle (or close ellipse)
            if aspect_ratio > 0.8 and aspect_ratio < 1.2:  # Allow for slight variation in a perfect circle
                return center

def get_target_center(target):
    image = load_image(1,11,"test")
    target_center = (400,300)
    # neu bia tron
    target_center = get_circle_target_center(image)
    # new bia elip
    target_center = get_elipse_target_center(image)
    return target_center

def calculate_point(lane, turn):
    # get the bullet holes from lane, turn, target
    holes = get_bullet_holes(lane, turn)
    (h,k) = get_target_center("test")
    #for bullet hole in bullet holes
    a = 7 #vertical of the smallest elipse
    b = 5 #horizontal of the smallest elipse

    score = 0
    for (x, y) in holes:
        if is_point_inside_ellipse(x,y,h,k,a,b):
            score = score + 10
        elif is_point_inside_ellipse(x,y,h,k,2*a,2*b):
            score = score + 9
        elif is_point_inside_ellipse(x,y,h,k,3*a,3*b):
            score = score + 8
        elif is_point_inside_ellipse(x,y,h,k,4*a,4*b):
            score = score + 7
        elif is_point_inside_ellipse(x,y,h,k,5*a,5*b):
            score = score + 6
        elif is_point_inside_ellipse(x,y,h,k,6*a,6*b):
            score = score + 5
        elif is_point_inside_ellipse(x,y,h,k,7*a,7*b):
            score = score + 4
        elif is_point_inside_ellipse(x,y,h,k,8*a,8*b):
            score = score + 3
        elif is_point_inside_ellipse(x,y,h,k,9*a,9*b):
            score = score + 2
        elif is_point_inside_ellipse(x,y,h,k,10*a,10*b):
            score = score + 1
        else:
            continue
    print(f"score: {score}")
    # return diem tong


def add_text(x,y,text, image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)
    thickness = 2
    cv2.putText(image, str(text), (x, y), font, font_scale, color, thickness)

array = []
results = []
def on_image_click(event, canvas, img, text_entries, lane, turn, target_name):
    image = cv2.imread(f"./Images/Result/Lane{lane}/{target_name}-{lane}-{turn}-marked.jpg")
    x, y = event.x, event.y
    print(x,y)
    hole = (x,y)
    # add pos x y to array
    array.append((x,y))
    for result in results:
        if result["name"] == f"{lane}-{turn}":
            result["holes"].append(hole)
            
    for (x,y) in array:
        add_text(x, y, turn, image)

    save_image(image, lane, turn, target_name)

def save_image(image, lane, turn, target_name):
    """Save the processed image to disk with dynamic target name."""
    cv2.imwrite(f"./Images/Result/Lane{lane}/{target_name}-{lane}-{turn}-marked.jpg", image)

def detect_bullet_hole(image, turn_num, lane):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blur, threshold1=50, threshold2=150)

    circles = cv2.HoughCircles(
        edges, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20, 
        param1=50, 
        param2=10, 
        minRadius=4, 
        maxRadius=8
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        valid_circles = []
        holes = []
        for circle in circles:
            x, y, r = circle
            hole = (x,y)
            valid_circles.append(circle)
            holes.append(hole)
            # add the holes detected to 
        result = {"name": f"{lane}-{turn_num}",
                  "lane": lane,
                  "turn": turn_num,
                  "holes": holes
                  }
        results.append(result)

        for (x, y, r) in valid_circles:
            top_left = (x - r, y - r)
            bottom_right = (x + r, y + r)
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 0, 255)
            thickness = 2
            cv2.putText(image, str(turn_num), (x - 2 * r // 3, y + 2 * r // 3), font, font_scale, color, thickness)

    save_image(image, 1, 1, "test")
    calculate_point(lane, turn_num)
    # Display the image
    image_path = f"./Images/Result/Lane1/test-1-1-marked.jpg"
    img = Image.open(image_path)
    tk_image = ImageTk.PhotoImage(img)

    root = tk.Toplevel()
    root.geometry("800x600")
    canvas = tk.Canvas(root, width=tk_image.width(), height=tk_image.height())
    canvas.pack()

    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    canvas.image = tk_image  # Keep a reference to avoid garbage collection

    text_entries = []

    # Bind the image click event to allow adding text
    canvas.bind("<Button-1>", lambda event: on_image_click(event, canvas, tk_image, text_entries, 1, 1, "test"))

    root.mainloop()
