import image_processing
import cv2
import tkinter as tk
import numpy as np
import math
import matplotlib.pyplot as plt
from tkinter import ttk
import camera

# detect bullet hole params - Bia So 4 - with basic setup cọc mắc màn TODO: define set up voi tripod
b_debug = False # if True, show debug images, edge, thresh, matches, remove bg, draw debug ellipses
thresh_value = 50

min_h_w = 6
max_h_w = 50
min_bullet_hole_area = 45
max_bullet_hole_area = 150
hole_to_hole_distance = 50 
min_hole_circularity = 0.3 #phu thuoc camera angle and distance to target
min_hole_ratio = 0.5
max_hole_ratio = 2

# ellipse detection params
min_ratio = 0.5
max_ratio = 1.2
min_ellipse_area = 2000000
max_ellipse_area = 3000000
min_angle = 80
max_angle = 360

delta_k = 9
delta_a = 0.5

results = []


def load_target_params(target):
    global thresh_value, min_h_w, max_h_w, min_bullet_hole_area, max_bullet_hole_area, min_hole_circularity, min_hole_ratio, max_hole_ratio, hole_to_hole_distance
    global min_ratio, max_ratio, min_ellipse_area, max_ellipse_area, min_angle, max_angle
    if target == "BiaSo4":
        
        # bullet detection params
        thresh_value = 50

        min_h_w = 5 #14
        max_h_w = 25 #20

        min_bullet_hole_area = 100 #139
        max_bullet_hole_area = 200 #201
        min_hole_circularity = 0.3 # 0.41
        min_hole_ratio = 0.5 # 0.9
        max_hole_ratio = 1.5 # 1.21
        hole_to_hole_distance = 50 # deprecated

        # ellipse detection params
        min_ratio = 1
        max_ratio = 1.2

        min_ellipse_area = 300000
        max_ellipse_area = 400000

        min_angle = 80
        max_angle = 100
        print("loaded")
    elif target == "BiaSo7":
        # bullet detection params
        thresh_value = 50

        min_h_w = 5 #8
        max_h_w = 25 #15

        min_bullet_hole_area = 25 #35
        max_bullet_hole_area = 110 #102
        min_hole_circularity = 0.3 # 0.59
        min_hole_ratio = 0.5 # 0.9
        max_hole_ratio = 1.5 # 1.36
        hole_to_hole_distance = 50 # deprecated

        # ellipse detection params
        min_ratio = 1
        max_ratio = 2

        min_ellipse_area = 50000
        max_ellipse_area = 9000000

        min_angle = 0
        max_angle = 360

def update_variables():
    global blur_value, adaptive_thresh_value, binary_thresh_value, edge_lower_value, edge_higher_value
    
    blur_value = blur_slider.get()
    adaptive_thresh_value = adaptive_thresh_slider.get()
    binary_thresh_value = binary_thresh_slider.get()
    edge_lower_value = edge_lower_slider.get()
    edge_higher_value = edge_higher_slider.get()
    
    # Update the result label to display the updated values
    result_label.config(text=f"Updated Values:\nBlur: {blur_value}\nAdaptive Threshold: {adaptive_thresh_value}\nBinary Threshold: {binary_thresh_value}\n"
                            f"Edge Lower: {edge_lower_value}\nEdge Higher: {edge_higher_value}\n\n")

# Create a new Toplevel window to edit variables
def open_variable_editor():
    global blur_slider, adaptive_thresh_slider, binary_thresh_slider, edge_lower_slider, edge_higher_slider, result_label
    
    # Create a new Toplevel window
    top = tk.Toplevel()
    top.title("Global Variables Editor")
    
    # Create a canvas widget
    canvas = tk.Canvas(top)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create a scrollbar widget
    scrollbar = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill="y")

    # Configure the canvas to work with the scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Create a frame inside the canvas to contain all widgets
    canvas_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=canvas_frame, anchor="nw")

    # Add all the sliders (trackbars) inside this frame
    ttk.Label(canvas_frame, text="Blur Value:").pack(pady=5)
    blur_slider = tk.Scale(canvas_frame, from_=0, to=20, orient="horizontal")
    blur_slider.set(blur_value)
    blur_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Adaptive Threshold Value:").pack(pady=5)
    adaptive_thresh_slider = tk.Scale(canvas_frame, from_=0, to=20, orient="horizontal")
    adaptive_thresh_slider.set(adaptive_thresh_value)
    adaptive_thresh_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Binary Threshold Value:").pack(pady=5)
    binary_thresh_slider = tk.Scale(canvas_frame, from_=0, to=255, orient="horizontal")
    binary_thresh_slider.set(binary_thresh_value)
    binary_thresh_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Edge Lower Value:").pack(pady=5)
    edge_lower_slider = tk.Scale(canvas_frame, from_=0, to=255, orient="horizontal")
    edge_lower_slider.set(edge_lower_value)
    edge_lower_slider.pack(pady=5)

    ttk.Label(canvas_frame, text="Edge Higher Value:").pack(pady=5)
    edge_higher_slider = tk.Scale(canvas_frame, from_=0, to=255, orient="horizontal")
    edge_higher_slider.set(edge_higher_value)
    edge_higher_slider.pack(pady=5)

    # Apply Button
    apply_button = ttk.Button(canvas_frame, text="Apply Variables", command=update_variables)
    apply_button.pack(pady=20)

    # Label to show updated values
    result_label = ttk.Label(canvas_frame, text=f"Current Values:\nBlur: {blur_value}\nAdaptive Threshold: {adaptive_thresh_value}\nBinary Threshold: {binary_thresh_value}\n"
                                               f"Edge Lower: {edge_lower_value}\nEdge Higher: {edge_higher_value}\n\n")
    result_label.pack(pady=10)

    # Update scroll region whenever the content changes
    canvas_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    top.geometry("350x600")


# check functions
def is_hole_inside_ellipse(x, y, h, k, a, b, angle):
    # Convert angle to radians
    angle_rad = math.radians(angle)
    
    # Translate the point to the ellipse's center (if necessary)
    x_translated = x - h
    y_translated = y - k
    
    # Rotate the point back by -angle
    x_rot = x_translated * math.cos(angle_rad) + y_translated * math.sin(angle_rad)
    y_rot = -x_translated * math.sin(angle_rad) + y_translated * math.cos(angle_rad)
    if (a == 0 or b == 0):
        return
    # Check if the point is inside the ellipse
    if (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1:
        return True
    else:
        return False

def is_hole_already_exist(lane, target, x, y, w, h):
    for result in results:
        if (result["lane"] == lane and result["target"] == target):
            for i, hole in enumerate(result["holes"]):  # Use enumerate to get the index
                (a, b, c, d) = hole
                # Coordinates of the two scores
                hole1 = np.array([x, y])
                hole2 = np.array([a, b])

                # Calculate Euclidean distance
                distance = np.linalg.norm(hole2 - hole1)
                if distance < hole_to_hole_distance:
                    if (w*h >= c*d):
                        # Hole exists, update the hole
                        result["holes"][i] = (x, y, w, h)  # Update the hole at index i
                        return True
                    else:
                        return True
    return False

def circularity_check(image,x, y, r):
    # Crop the region around the circle
    cropped_circle = image[y-r:y+r, x-r:x+r]
    
    # Create a mask for the circle (ensure it is single-channel for contouring)
    mask = np.zeros(cropped_circle.shape, dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)  # Full circle mask
    
    # Get the intensity values inside and outside the circle
    inside_values = cropped_circle[mask == 255]
    outside_values = cropped_circle[mask == 0]
    
    # Calculate contrast: mean intensity inside vs outside
    inside_mean = np.mean(inside_values)
    outside_mean = np.mean(outside_values)
    
    contrast = abs(inside_mean - outside_mean) / max(inside_mean, outside_mean)
    
    # Convert the mask to grayscale (it should already be single-channel, but for safety)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) > 2 else mask
    
    # Calculate circularity using the contour of the circle
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = contours[0]
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if perimeter == 0:
            return False
        
        # Circularity formula
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # Check if the circularity is within the acceptable range and contrast threshold
        if 0.5 <= circularity <= 1.1:
            return True
    
    return True

# getters
def get_bullet_holes(lane, turn, target):
    for result in results:
        if result["name"] == f"{lane}-{turn}-{target}":
            return result["holes"]


def get_center_ellipse_parameters(image, target):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # Use edge detection (Canny)
    edges = cv2.Canny(adaptive_thresh, 150, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center_ellipse = None
    min_aspect_ratio = float("inf")

    # Iterate over the contours and fit ellipses
    for contour in contours:
        if len(contour) >= 5:  # Fit an ellipse requires at least 5 points
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            (h, k) = ellipse[0]  # Center of the ellipse
            (a, b) = ellipse[1]  # Major and minor axes
            angle = ellipse[2]  # Angle of rotation of the ellipse
            # Check if any parameter is NaN or invalid
            if np.isnan(h) or np.isnan(k) or np.isnan(a) or np.isnan(b) or np.isnan(angle):
                continue
            # Calculate the area of the ellipse
            area = np.pi * a * b

            if area == 0 or a == 0 or b == 0:
                continue

            # Calculate aspect ratio
            aspect_ratio = max(a, b) / min(a, b)

            # Apply aspect ratio and area filters
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                continue
            if area < min_ellipse_area or area > max_ellipse_area:
                continue
            if angle < min_angle or angle > max_angle:
                continue
            if aspect_ratio < min_aspect_ratio:
                # If this ellipse passes the criteria, save it
                center_ellipse = ellipse
                min_aspect_ratio = aspect_ratio

                # Draw the ellipse on the image (for visualization purposes)
                #cv2.ellipse(image, (int(h), int(k)), (int(a), int(b)), angle, 0, 360, (0, 255, 0), 2)  # Green color, thickness 2
            print(f"Found ellipse: Area={area}, Aspect Ratio={aspect_ratio}, Angle={angle}, Position=({h}, {k} Axes=({a//4}, {b//4})")
    

    # Get image dimensions
    height, width = image.shape[:2]

    # Default values for ellipse parameters in case no ellipse is found
    h, k = int(width // 2), int(height // 2)

    if target == "BiaSo4":
        a, b, h, k, angle = 74, 84, 965, 484, 90 #default if no ellipse found
    elif target == "BiaSo7":
        a, b, h, k, angle = 74, 50, h, k, 90 #default if no ellipse found

    if center_ellipse:
        (h, k) = center_ellipse[0]  # Center
        (a, b) = center_ellipse[1]  # Axes
        angle = center_ellipse[2]   # Angle
        a = a // 4  # Scale down major axis for your use case
        b = b // 4  # Scale down minor axis for your use case
    cv2.ellipse(image, (int(h), int(k)), (int(a), int(b)), angle, 0, 360, (0,0, 255), 2)  
    return int(a), int(b), int(h), int(k), int(angle)




# draw
def draw_debug_elipse(image, a, b, h, k, angle, target):
    
    # rotation angle in degrees
    start_angle = 0  # starting angle of the arc
    end_angle = 360  # ending angle of the arc (full ellipse)
    if target == "BiaSo4":
        for i in range(1,6): # bia so 4
        # Draw the ellipse on the image
            if i == 1: 
                k -= 5*i
                a += 5*i
                b += 5*i
                cv2.circle(image, (int(h),int(k-5)), 1, (0,0,255), 2)
            elif  i > 1:
                k += 2*i
                a += 0.5*i
                b += 0.5*i
            if i >= 3:
                k += 2
                a -= 0.5*i
                b -= 0.5*i
            cv2.ellipse(image, (int(h),int(k)), (int(a)*i,int(b)*i), angle, start_angle, end_angle, (0, 255, 0), 2)

def calculate_score(holes, lane, turn, target):
    try:
        image = image_processing.load_image(lane, turn, target)
    except Exception as e:
        print(e)
        return f"Kiểm tra lại kết nối camera dải số {lane} mục tiêu {target}"
    
    if holes == None:
        print(f"Loạt {turn}, bệ số {lane} Mục tiêu {target} an toàn")
        return f"Loạt {turn}, bệ số {lane} Mục tiêu {target} an toàn"
    try:
        (a,b,h,k,angle) = get_center_ellipse_parameters(image, target)
    except Exception as e:
        print(e)
        return f"Không thể xác định elipse trung tâm"
    total_score = 0
    score = 0
    scores = []
    i = 1
    message = f"Loạt {turn}, bệ số {lane}, mục tiêu {target}:"
    
    for (x, y, w, j) in holes:
        if i==1:
            k -= 18*i
        if i>=3:
            k += 9*i
            a += 0.5*i
            b += 0.5*i
        
        if is_hole_inside_ellipse(x,y,h,k,a,b,angle):
            score = 10
        elif is_hole_inside_ellipse(x,y,h,k,2*a,2*b,angle):
            score = 9
        elif is_hole_inside_ellipse(x,y,h,k,3*a,3*b,angle):
            score = 8
        elif is_hole_inside_ellipse(x,y,h,k,4*a,4*b,angle):
            score = 7
        elif is_hole_inside_ellipse(x,y,h,k,5*a,5*b,angle):
            score = 6
        elif is_hole_inside_ellipse(x,y,h,k,6*a,6*b,angle):
            score = 5
        elif is_hole_inside_ellipse(x,y,h,k,7*a,7*b,angle):
            score = 4
        elif is_hole_inside_ellipse(x,y,h,k,8*a,8*b,angle):
            score = 3
        elif is_hole_inside_ellipse(x,y,h,k,9*a,9*b,angle):
            score = 2
        elif is_hole_inside_ellipse(x,y,h,k,10*a,10*b,angle):
            score = 1
        else:
            score = 0
            #continue
        
        result = f"\n Phát {i}: {score} điểm"
        message = message + result
        total_score = total_score + score
        scores.append(score)
        i += 1
    
    message = message + f"\n Tổng: {total_score} điểm"
    print(message)

    return total_score, message

def compare_and_detect(lane, turn, target):
    load_target_params(target)
    try:
        image_prev_turn = image_processing.load_image(lane, turn-1, target)
        image_curr_turn = image_processing.load_image(lane, turn, target)
        processed_prev,_,_,_ = image_processing.process_image(image_prev_turn)# process xong nhan them 2 lo, trong do co 1 lo sai?
        processed_curr,_,_,_ = image_processing.process_image(image_curr_turn)
        p_gamma = image_processing.gamma_correction(image_prev_turn)
        c_gamma = image_processing.gamma_correction(image_curr_turn)
        #z_prev = image_processing.remove_background(processed_prev)
        #z_curr = image_processing.remove_background(processed_curr)
        gray_prev = image_processing.preprocess_image(processed_prev)
        gray_curr = image_processing.preprocess_image(processed_curr) # giảm ảnh hưởng của nắng

    except Exception as e:
        print(e)
        return None, None
    
    # Step 1: Feature detection and matching (ORB in this case)
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)

    # Use BFMatcher to find the best matches between the descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Sort the good matches based on distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Step 3: Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    try:
        # Step 4: Calculate homography matrix to align the images using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
    except Exception as e:
        print(e)
        return None, None
    # Step 5: Warp the 'after' image to align with the 'before' image
    c_height, c_width = gray_curr.shape
    try:
        aligned_before = cv2.warpPerspective(gray_prev, M, (c_width, c_height))
    except Exception as e:
        print(e)
        return None, None
    # Step 6: Calculate the absolute difference between the images
    diff_image = cv2.absdiff(aligned_before, gray_curr)

    # Step 7: Threshold the difference image to highlight changes
    _, thresh_diff = cv2.threshold(diff_image, 100, 255, cv2.THRESH_BINARY)

    # Step 8: Find contours in the thresholded difference image
    contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Step 9: Draw bounding boxes and annotate with numbers
    valid_holes = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # filtering holes
        if is_hole_already_exist(lane, target, x,y,w,h):
            continue
        if w < min_h_w or w > max_h_w:
            continue
        if h < min_h_w or h > max_h_w:
            continue

        # continue filtering holes
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        contour_area = cv2.contourArea(contour)
        aspect_ratio = w / h if h != 0 else 0
        circularity = 4 * np.pi * contour_area / (perimeter ** 2)
        
        if contour_area < min_bullet_hole_area or contour_area > max_bullet_hole_area:
            continue
        if circularity < min_hole_circularity:
            continue
        if min_hole_ratio > aspect_ratio or aspect_ratio > max_hole_ratio:
            continue
        #print(perimeter, contour_area, circularity, aspect_ratio)
        print(x,y,w,h, contour_area, aspect_ratio, circularity)
        # pass all the check? then append to valid_holes
        valid_holes.append((x, y, w, h))
    # construct result
    total_score, result_text = calculate_score(valid_holes, lane, turn, target)
    result = {"name": f"{lane}-{turn}-{target}",
            "lane": lane,
            "turn": turn,
            "target": target,
            "holes": valid_holes,
            "total_score": total_score,
            "result_text": result_text
            }
                    
    results.append(result)
    
    # draw bounding boxes and annotate with numbers
    for (x,y,w,h) in valid_holes:
        cv2.rectangle(image_curr_turn, (x, y), (x + w, y + h), (0, 255, 0), 1)
        text = f"Loat {turn}"
        cv2.putText(image_curr_turn, text, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        

    image_processing.save_image(image_curr_turn, lane, turn, target)
    # Step 10: Show the images with bounding boxes and numbers (debug only)
    if False:
        # Show the matched keypoints
        image_matches = cv2.drawMatches(gray_prev, kp1, gray_curr, kp2, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(image_matches)
        plt.title('Feature Matches')
        plt.show()
        try:
            (a, b, h, k, angle) = get_center_ellipse_parameters(image_curr_turn, target)
            draw_debug_elipse(image_curr_turn,a, b, h, k, angle, target)
        except Exception as e:
            print(e)
        cv2.imshow('Bullet Holes Detected', image_curr_turn)
        cv2.imshow('Aligned Before Image', aligned_before)
        cv2.imshow('Difference Image', diff_image)
        cv2.imshow('Thresholded Difference (Bullet Hole)', thresh_diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image_curr_turn, result_text, total_score

if __name__=="__main__":
    b_debug = True
    #cam = camera.Camera(1,"BiaSo7", 1)
    #img = cv2.imread("./HinhAnh/DaiBan1/BiaSo4Test-1-1.jpg")
    
    #img = cam.capture_image(5)
    #img2 = cam.capture_image(9)
    #save_image(img,1,10,"BiaSo4")
    compare_and_detect(1, 6, "BiaSo4")
    #get_center_ellipse_parameters(img)