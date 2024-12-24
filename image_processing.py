import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from tkinter import messagebox

def load_image(lane, turn, target):
    image = cv2.imread(f'./Images/Lane{lane}/{target}-{lane}-{turn}.jpg')
    return image

def load_result(lane, turn, target):
    result = cv2.imread(f'./Images/Result/Lane{lane}/{target}-{lane}-{turn}-marked.jpg')
    return result

def otsu_thresholding(gray):
    # Apply adaptive thresholding
    # Apply Otsu's thresholding
    return cv2.threshold(
        gray, 
        0, 
        255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

def adaptive_thresholding(gray):
    # Apply adaptive thresholding
    return (1.0, cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 
        11,  # Block size (local region size)
        2    # Constant subtracted from mean or weighted mean
    ))

def is_score_inside_ellipse(x, y, h, k, a, b):
    """Check if a score (x, y) is inside an ellipse with center (h, k),
    semi-major axis a, and semi-minor axis b."""
    
    # Apply the ellipse equation
    equation_result = ((x - h) ** 2) / (a ** 2) + ((y - k) ** 2) / (b ** 2)
    # h,k: tam bia
    # a: canh doc
    # b: canh ngang
    # If the result is less than or equal to 1, the score is inside or on the ellipse
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
        if len(contour) >= 5:  # Fit an ellipse requires at least 5 scores
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

def draw_debug_elipse(image, a, b, h, k):
    # Parameters for the ellipse
    center = (h, k)  # center of the ellipse
    axes = (a, b)  # axes lengths (semi-major and semi-minor axes)
    angle = 90  # rotation angle in degrees
    start_angle = 0  # starting angle of the arc
    end_angle = 360  # ending angle of the arc (full ellipse)

    for i in range(1,10):
    # Draw the ellipse on the image
        cv2.ellipse(image, (h,k), (a*i,b*i), angle, start_angle, end_angle, (255, 0, 0), 1)
        cv2.circle(image, (h,k), 1, (0,0,255), 1)

def calculate_score(lane, turn):
    # get the bullet holes from lane, turn, target
    holes = get_bullet_holes(lane, turn)
    #(h,k) = get_target_center("test")
    h = 400
    k = 300
    #for bullet hole in bullet holes
    a = 100 #vertical of the smallest elipse
    b = 80 #horizontal of the smallest elipse

    total_score = 0
    score = 0
    scores = []
    i = 0
    message = f"Loạt {turn}, bệ số {lane}:"
    for (x, y, r) in holes:
        i = i + 1
        if is_score_inside_ellipse(x,y,h,k,a,b):
            score = 10
        elif is_score_inside_ellipse(x,y,h,k,2*a,2*b):
            score = 9
        elif is_score_inside_ellipse(x,y,h,k,3*a,3*b):
            score = 8
        elif is_score_inside_ellipse(x,y,h,k,4*a,4*b):
            score = 7
        elif is_score_inside_ellipse(x,y,h,k,5*a,5*b):
            score = 6
        elif is_score_inside_ellipse(x,y,h,k,6*a,6*b):
            score = 5
        elif is_score_inside_ellipse(x,y,h,k,7*a,7*b):
            score = 4
        elif is_score_inside_ellipse(x,y,h,k,8*a,8*b):
            score = 3
        elif is_score_inside_ellipse(x,y,h,k,9*a,9*b):
            score = 2
        elif is_score_inside_ellipse(x,y,h,k,10*a,10*b):
            score = 1
        else:
            score = 0
            continue
        
        if turn == 1:
            score = score -1
        result = f"\n Phát {i}: {score} điểm"
        message = message + result
        total_score = total_score + score
        scores.append(score)
        print(f"phat dan thu {i}: {score} diem")
    print(f"tong so diem: {total_score} diem")
    
    message = message + f"\n Tổng: {total_score} điểm"
    messagebox.showinfo("Báo bia", message)
    # return diem tong


def add_text(x,y,text, image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)
    thickness = 2
    cv2.putText(image, str(text), (x, y), font, font_scale, color, thickness)

array = []
results = []
def on_image_click(event, canvas, img, text_entries, lane, turn, target):
    image = cv2.imread(f"./Images/Result/Lane{lane}/{target}-{lane}-{turn}-marked.jpg")
    x, y = event.x, event.y
    print(x,y)
    hole = (x,y,1)
    # add pos x y to array
    array.append((x,y))
    for result in results:
        if result["name"] == f"{lane}-{turn}":
            result["holes"].append(hole)
            
    for (x,y) in array:
        add_text(x, y, turn, image)

    #save_image(image, lane, turn, target)
    calculate_score(lane, turn)

def save_image(image, lane, turn, target):
    """Save the processed image to disk with dynamic target name."""
    cv2.imwrite(f"./Images/Result/Lane{lane}/{target}-{lane}-{turn}-marked.jpg", image)

def is_hole_already_exist(x,y,r):
    for result in results:
        for (a,b,c) in result["holes"]:
            # Coordinates of the two scores
            hole1 = np.array([x, y])
            hole2 = np.array([a, b])

            # Calculate Euclidean distance
            distance = np.linalg.norm(hole2 - hole1)
            if distance < 10:
                # hole existed
                print("hole exist")
                return True
    return False

def draw_debug(image, x,y,r, turn):
    top_left = (x - r, y - r)
    bottom_right = (x + r, y + r)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)
    thickness = 2
    cv2.putText(image, str(turn), (x - 2 * r // 3, y + 2 * r // 3), font, font_scale, color, thickness)

def circularity_check(x, y, r):
    area = np.pi*r**2
    perimeter = 2*np.pi*r

    if perimeter > 0:  # Avoid division by zero
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return circularity

def detect_bullet_hole(image, turn, lane, target):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    #_, thresh = adaptive_thresholding(gray_blurred)
    edges = cv2.Canny(gray_blurred, threshold1=50, threshold2=150)
    
    circles = cv2.HoughCircles(
        edges, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=100, 
        param1=50, 
        param2=10, 
        minRadius=3, 
        maxRadius=9
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        valid_circles = []
        holes = []
        # luu ket qua
        for circle in circles:
            x, y, r = circle
            hole = (x,y,r)
            print(circularity_check(x,y,r))

            # check xem hole nay co trung voi loat truoc khong
            if not is_hole_already_exist(x,y,r):
                valid_circles.append(circle)
                holes.append(hole)

        result = {"name": f"{lane}-{turn}",
                  "lane": lane,
                  "turn": turn,
                  "holes": holes
                  }
        results.append(result)

        for result in results:
            print(f"loat {result["turn"]} ban trung : {len(result["holes"])} phat dan")
            for (x,y,r) in result["holes"]:
                draw_debug(image, x,y,r,result["turn"])
            
        
    
    save_image(image, lane, turn, target)
    # Display the image
    image_path = f"./Images/Result/Lane{lane}/{target}-{lane}-{turn}-marked.jpg"
    img = Image.open(image_path)
    tk_image = ImageTk.PhotoImage(img)

    root = tk.Toplevel()
    root.geometry("1920x1080")
    canvas = tk.Canvas(root, width=tk_image.width(), height=tk_image.height())    
    canvas.pack()

    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    canvas.image = tk_image  # Keep a reference to avoid garbage collection
    calculate_score(lane, turn)

    text_entries = []

    # Bind the image click event to allow adding text
    canvas.bind("<Button-1>", lambda event: on_image_click(event, canvas, tk_image, text_entries, lane, turn, target))

    root.mainloop()


if __name__ == "__main__":
    image = cv2.imread("./Images/Lane1/test-1-3.jpg")
    overlay = cv2.imread("./Images/Lane1/overlay.jpg")
    #draw_debug_elipse(image, 100,80,400,300)
    # Apply the drawing (overlay it back onto the original image)
    # Blend the original image with the drawn image using alpha blending
    alpha = 0.5  # Transparency factor
    blended_image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


    cv2.imshow("debug", blended_image)
    #save_image(blended_image, 1,1, "test")
    cv2.imwrite("./Images/Lane1/test-1-3.jpg", blended_image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()