import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import camera
from tkinter import messagebox
import object_detection
import obj_detect


# tham so
array = []
results = []
#
def load_image(lane, turn, target):
    image = cv2.imread(f'./Images/Lane{lane}/{target}-{lane}-{turn}.jpg')
    return image

def load_result(lane, turn, target):
    result = cv2.imread(f'./Images/Result/Lane{lane}/{target}-{lane}-{turn}-marked.jpg')
    return result

def otsu_thresholding(gray):
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
        11    # Constant subtracted from mean or weighted mean
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

    for i in range(1,11):
    # Draw the ellipse on the image
        cv2.ellipse(image, (h,k), (a*i,b*i), angle, start_angle, end_angle, (255, 0, 0), 1)
        cv2.circle(image, (h,k), 1, (0,0,255), 1)

def calculate_score(lane, turn):
    # get the bullet holes from lane, turn, target
    holes = get_bullet_holes(lane, turn)
    #(h,k) = get_target_center("test")
    h = 958
    k = 750
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
        
        #if turn == 1:
        #    score = score -1
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

def is_hole_already_exist(x, y, r):
    for result in results:
        for i, hole in enumerate(result["holes"]):  # Use enumerate to get the index
            (a, b, c) = hole
            # Coordinates of the two scores
            hole1 = np.array([x, y])
            hole2 = np.array([a, b])

            # Calculate Euclidean distance
            distance = np.linalg.norm(hole2 - hole1)
            if distance < 20:
                # Hole exists, update the hole
                print("Hole exists, updating the hole value.")
                result["holes"][i] = (x, y, r)  # Update the hole at index i
                return True
    return False

def draw_debug(image, x,y,r, turn):
    #draw bounding box
    top_left = (x - r, y - r)
    bottom_right = (x + r, y + r)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 1)
    # draw turn number
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

def detect_bullet_hole(image, turn, lane, target, gray_blur_value ,thresh_value, edge_thresh_1, edge_thresh_2,  min_dist, param1, param2, min_rad, max_rad):
    # pre-processing
    draw_debug_elipse(image, 100, 80, 958, 750)
    cropped_target, target_masked, target_mask = object_detection.detect_target(image)
    zoomed = object_detection.zoom_in(cropped_target, 1)
    gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (gray_blur_value, gray_blur_value), 0)
    _, thresh_binary = cv2.threshold(gray_blurred, thresh_value, 255, cv2.THRESH_BINARY) # da test ok voi anh Tuan 9,200,150,150,100,150,11,1,11, van con bi mat phang nhap nho
    #_, thresh_a = adaptive_thresholding(gray_blurred)
    _, thresh_b = cv2.threshold(thresh_binary, thresh_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh = adaptive_thresholding(thresh_b) # used for various lighting condition
    edges = cv2.Canny(thresh, threshold1=edge_thresh_1, threshold2=edge_thresh_2)
    # find holes
    circles = cv2.HoughCircles(
        edges, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=min_dist, 
        param1=param1, 
        param2=param2, 
        minRadius=min_rad, 
        maxRadius=max_rad
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        valid_circles = []
        holes = []
        # luu ket qua
        for circle in circles:
            x, y, r = circle
            hole = (x,y,r)
            print(r)
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
                draw_debug(zoomed, x,y,r,result["turn"])
            
    cv2.imshow('Video Frame', zoomed)
    cv2.imshow("edge", edges)
    calculate_score(lane, turn)
    save_image(zoomed, lane, turn, target)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """if not __name__ == "__main__":
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
        text_entries = []

        # Bind the image click event to allow adding text
        canvas.bind("<Button-1>", lambda event: on_image_click(event, canvas, tk_image, text_entries, lane, turn, target))

        root.mainloop()"""

def nothing(x):
    pass
if __name__ == "__main__":
    cam = camera.Camera(1,"BiaTest", 1)
    img = cam.capture_image(1)
    # pre-processing
    cropped_target, target_masked, target_mask = object_detection.detect_target(img)
    zoomed = object_detection.zoom_in(cropped_target, 1)
    gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(gray_blurred, 150, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, threshold1=150, threshold2=150)
    #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find holes
    circles = cv2.HoughCircles(
        edges, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=100, 
        param1=150, 
        param2=15, 
        minRadius=3, 
        maxRadius=11
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        valid_circles = []
        holes = []
        # luu ket qua
        for circle in circles:
            x, y, r = circle
            hole = (x,y,r)
            print(x, y, r)
            print(circularity_check(x,y,r))
            # check xem hole nay co trung voi loat truoc khong
            if not is_hole_already_exist(x,y,r):
                valid_circles.append(circle)
                holes.append(hole)

        result = {"name": f"{1}-{1}",
                "lane": 1,
                "turn": 1,
                "holes": holes
                }
        results.append(result)

        for result in results:
            #print(f"loat {result["turn"]} ban trung : {len(result["holes"])} phat dan")
            for (x,y,r) in result["holes"]:
                draw_debug(zoomed, x,y,r,result["turn"])

    # Display the frame
    cv2.imshow('Video Frame', zoomed)
    cv2.waitKey(0)
    # Break the loop when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

        #FIND ELIPSE
    # Step 5: Detect bullet holes (circles/ellipses)
    """detected_bullet_holes = []
    for contour in contours:
        if 1<cv2.contourArea(contour) <10:  # Filter out small contours
            # Step 6: Fit ellipse to contour (handles perspective distortion)
            if len(contour) >= 5:  # At least 5 points required to fit an ellipse
                ellipse = cv2.fitEllipse(contour)
                detected_bullet_holes.append(ellipse)
            else:
                print("not elipse")
    for ellipse in detected_bullet_holes:
        center, axes, angle = ellipse
        
        # Ensure center is a tuple of integers
        center = tuple(map(int, center))
        
        # Convert axes to integers (semi-major and semi-minor axes)
        axes = tuple(map(int, axes))
        
        color = (0, 255, 0)  # Green color
        thickness = 2
        
        # Draw the ellipse
        cv2.ellipse(img, center, axes, angle, 0, 360, color, thickness)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """