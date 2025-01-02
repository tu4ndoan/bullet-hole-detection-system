import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from tkinter import messagebox
import object_detection
import math
import matplotlib.pyplot as plt


# tham so
array = []
results = []
result_texts = []
#
def load_image(lane, turn, target):
    image = cv2.imread(f'./HinhAnh/DaiBan{lane}/{target}-{lane}-{turn}.jpg', 1)
    return image

def load_result(lane, turn, target):
    result = cv2.imread(f'./HinhAnh/KetQua/DaiBan{lane}/{target}-{lane}-{turn}-marked.jpg')
    return result

def save_image(image, lane, turn, target):
    print(lane, turn, target)
    #TODO:check dir exist and create
    cv2.imwrite(f"./HinhAnh/KetQua/DaiBan{lane}/{target}-{lane}-{turn}-marked.jpg", image)

def is_hole_inside_ellipse(x, y, h, k, a, b, angle):
    # Convert angle to radians
    angle_rad = math.radians(angle)
    print(x,y)
    
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

def get_bullet_holes(lane, turn):
    for result in results:
        if result["name"] == f"{lane}-{turn}":
            return result["holes"]

def get_center_ellipse_parameters(image):
    # Convert to grayscale
    cropped_target, target_masked, target_mask = object_detection.detect_target(image)
    zoomed = object_detection.zoom_in(cropped_target, 1)
    #cv2.imshow("zoom", zoomed)
    #cv2.waitKey(0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram Equalization to improve contrast
    gray_equalized = cv2.equalizeHist(gray)
    
    # Apply Gaussian Blur to smooth lighting variations
    gray_blurred = cv2.GaussianBlur(gray_equalized, (15, 15), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to the image
    clahe_image = clahe.apply(gray_blurred)
    # Adaptive thresholding to account for varying lighting conditions
    thresh_adaptive = cv2.adaptiveThreshold(
        gray_blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
   
    # Further binary thresholding if needed
    #ret, inv_thresh = cv2.threshold(gray_blurred, 150, 255, cv2.THRESH_BINARY_INV)
    #_, thresh = cv2.threshold(gray_blurred, 200, 255, cv2.THRESH_BINARY)
    #ret, trunc_thresh = cv2.threshold(clahe_image, 55, 255, cv2.THRESH_TRUNC)
    ret, otsu_thresh = cv2.threshold(thresh_adaptive, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Edge detection with dynamic thresholds
    median_intensity = np.median(gray_blurred)
    lower_thresh = max(0, median_intensity - 50)
    upper_thresh = min(255, median_intensity + 150)
    edges = cv2.Canny(otsu_thresh, threshold1=lower_thresh, threshold2=upper_thresh)
    
    
    # Apply dilation and erosion to link fragmented edges
    kernel = np.ones((3, 3), np.uint8)  # A 3x3 kernel for dilation and erosion
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)  # Dilate to join edges
    linked_edges = cv2.erode(dilated_edges, kernel, iterations=1)  # Erode to reduce noise

    # Find contours
    contours, _ = cv2.findContours(linked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_ellipse = None

    
    # Iterate over the contours and fit ellipses
    for contour in contours:
        if len(contour) >= 5:  # Fit an ellipse requires at least 5 scores
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            (h, k) = ellipse[0]
            (a, b) = ellipse[1]
            # Extract ellipse parameters (center, axes, angle)
            center, axes, angle = ellipse
            area = np.pi * a * b
            aspect_ratio = 0
            if (min(axes) != 0):
                # Compute the aspect ratio (major axis / minor axis) to check if it's "perfect"
                aspect_ratio = max(axes) / min(axes)

            # If the aspect ratio is close to 1, it is more likely a perfect circle (or close ellipse)
            
            #if (4000 < cv2.contourArea(contour) < 8000):
             #   center_ellipse = ellipse
                #print(f"found elipse {cv2.contourArea(contour)} {aspect_ratio} {angle}")
                # Allow for slight variation in a perfect circle
                #cv2.ellipse(image, ellipse, (0, 255, 0), 2)
    #cv2.putText(image, str(), (1030, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.circle(image, (935, 515), 2, (255,0,0), 2)
    a, b, h, k, angle = 0, 0, 0, 0, 0
    if (center_ellipse):
        (h, k) = center_ellipse[0]
        (a, b) = center_ellipse[1]
        angle = center_ellipse[2]
    #draw_debug_elipse(image, 125, 145, 880, 495, 90)
    cv2.imshow("elipse", image)
    cv2.imshow("edge", linked_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 125, 145, 935, 515, 90

def draw_debug_elipse(image, a, b, h, k, angle):
    # rotation angle in degrees
    start_angle = 0  # starting angle of the arc
    end_angle = 360  # ending angle of the arc (full ellipse)
    for i in range(1,11):
    # Draw the ellipse on the image
        if i >= 3:
            k += 10*i
        cv2.ellipse(image, (int(h),int(k)), (int(a)*i,int(b)*i), angle, start_angle, end_angle, (0, 255, 0), 2)
        cv2.circle(image, (int(h),int(k)), 1, (0,0,255), 1)

def calculate_score(lane, turn, target):
    # get the bullet holes from lane, turn, target
    holes = get_bullet_holes(lane, turn)
    if holes == None:
        return
    image = load_image(lane, turn, target)
    if not image.any(): # load image fail
        return
    (a,b,h,k,angle) = get_center_ellipse_parameters(image)
    total_score = 0
    score = 0
    scores = []
    i = 0
    message = f"Loạt {turn}, bệ số {lane}:"
    
    print(len(holes))
    for (x, y, w, j) in holes:
        if i >= 3:
            k += 10*i
        i = i + 1
        
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
            continue
        
        result = f"\n Phát {i}: {score} điểm"
        message = message + result
        total_score = total_score + score
        scores.append(score)
        print(f"phat dan thu {i}: {score} diem")
    print(f"tong so diem: {total_score} diem")
    
    message = message + f"\n Tổng: {total_score} điểm"
    messagebox.showinfo("Báo bia", message)
    # return diem tong
    return message

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
        cv2.putText(img, "text", (x, y), 1, 1, (255, 0, 0), 1)

    #save_image(image, lane, turn, target)
    calculate_score(lane, turn)

def is_hole_already_exist(x, y, w, h):
    for result in results:
        for i, hole in enumerate(result["holes"]):  # Use enumerate to get the index
            (a, b, c, d) = hole
            # Coordinates of the two scores
            hole1 = np.array([x, y])
            hole2 = np.array([a, b])

            # Calculate Euclidean distance
            distance = np.linalg.norm(hole2 - hole1)
            if distance < 50:
                if (w*h >= c*d):
                    # Hole exists, update the hole
                    print("Hole exists, updating the hole value.")
                    result["holes"][i] = (x, y, w, h)  # Update the hole at index i
                    return True
                else:
                    print("hole smaller")
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

def circularity_check(image,x, y, r):
    """
    Checks if a detected circle is likely a bullet hole based on circularity and contrast.
    
    Args:
    - image (np.array): The input image.
    - x (int): The x-coordinate of the circle's center.
    - y (int): The y-coordinate of the circle's center.
    - r (int): The radius of the circle.
    - min_contrast (float): Minimum contrast ratio between the inside and outside of the circle.
    - max_circularity (float): Maximum allowed circularity for a valid bullet hole.
    
    Returns:
    - bool: True if the circle passes the circularity and contrast checks, False otherwise.
    """
    
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

def detect_bullet_hole_test(image, turn, lane, target, gray_blur_value, block_size, thresh_a_val ,thresh_value, edge_thresh_1, edge_thresh_2,  min_dist, param1, param2, min_rad, max_rad):
    # pre-processing
    #draw_debug_elipse(image, 100, 80, 958, 750)
    cropped_target, target_masked, target_mask = object_detection.detect_target(image)
    zoomed = object_detection.zoom_in(cropped_target, 1) #TODO check xem tại sao crop sai ban đêm
    zoomed = image
    gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
    
    # Histogram Equalization to improve contrast
    gray_equalized = cv2.equalizeHist(gray)
    
    # Apply Gaussian Blur to smooth lighting variations
    gray_blurred = cv2.GaussianBlur(gray_equalized, (gray_blur_value, gray_blur_value), 0)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to the image
    #clahe_image = clahe.apply(gray_blurred)
    # Adaptive thresholding to account for varying lighting conditions
    thresh_adaptive = cv2.adaptiveThreshold(
        gray_blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 
        thresh_a_val
    )
    
   
    # Further binary thresholding if needed
    #ret, inv_thresh = cv2.threshold(gray_blurred, 150, 255, cv2.THRESH_BINARY_INV)
    #_, thresh = cv2.threshold(gray_blurred, 200, 255, cv2.THRESH_BINARY)
    #ret, trunc_thresh = cv2.threshold(clahe_image, 55, 255, cv2.THRESH_TRUNC)
    ret, otsu_thresh = cv2.threshold(thresh_adaptive, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Edge detection with dynamic thresholds
    median_intensity = np.median(gray_blurred)
    lower_thresh = max(0, median_intensity - edge_thresh_1)
    upper_thresh = min(255, median_intensity + edge_thresh_2)
    edges = cv2.Canny(otsu_thresh, threshold1=lower_thresh, threshold2=upper_thresh)
    
    
    # Apply dilation and erosion to link fragmented edges
    kernel = np.ones((3, 3), np.uint8)  # A 3x3 kernel for dilation and erosion
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)  # Dilate to join edges
    linked_edges = cv2.erode(dilated_edges, kernel, iterations=1)  # Erode to reduce noise
    
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        linked_edges, 
        cv2.HOUGH_GRADIENT, 
        dp=0.1, 
        minDist=min_dist, 
        param1=param1, 
        param2=param2, 
        minRadius=min_rad, 
        maxRadius=max_rad
    )

    # Process detected circles and filter based on size and circularity
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        valid_circles = []
        holes = []
        for circle in circles:
            x, y, r = circle
            if is_hole_already_exist(x, y, r):
                print("hole exist")
                continue
            print(r)
            if circularity_check(image, x, y, r) and min_rad <= r <= max_rad:
                valid_circles.append(circle)
                holes.append((x, y, r))

                result = {"name": f"{lane}-{turn}",
                        "lane": lane,
                        "turn": turn,
                        "holes": holes
                        }
        
                results.append(result)
        
        for result in results:
            print(f"loat {result['turn']} ban trung : {len(result['holes'])} phat dan")
    
            for (x, y, r) in result['holes']:
                draw_debug(zoomed, x, y, r, result["turn"])
    if turn != 0:
        cv2.imshow('Video Frame', zoomed)
        calculate_score(lane, turn)
        save_image(zoomed, lane, turn, target)  
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram Equalization to improve contrast
    gray_equalized = cv2.equalizeHist(gray)
    
    # Apply Gaussian Blur to smooth lighting variations
    gray_blurred = cv2.GaussianBlur(gray_equalized, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to the image
    clahe_image = clahe.apply(gray_blurred)
    # Adaptive thresholding to account for varying lighting conditions
    thresh_adaptive = cv2.adaptiveThreshold(
        clahe_image, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    # Further thresholding
    ret, otsu_thresh = cv2.threshold(thresh_adaptive, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Edge detection with dynamic thresholds
    median_intensity = np.median(gray_blurred)
    lower_thresh = max(0, median_intensity - 150)
    upper_thresh = min(255, median_intensity + 150)
    edges = cv2.Canny(otsu_thresh, threshold1=lower_thresh, threshold2=upper_thresh)
    
    
    # Apply dilation and erosion to link fragmented edges
    kernel = np.ones((3, 3), np.uint8)  # A 3x3 kernel for dilation and erosion
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)  # Dilate to join edges
    linked_edges = cv2.erode(dilated_edges, kernel, iterations=1)  # Erode to reduce noise

    return linked_edges

def compare_and_detect(lane, turn, target):
    print(f"compare and detect {lane} {turn} {target}")
    image_prev_turn = load_image(lane, turn-1, target)
    image_curr_turn = load_image(lane, turn, target)
    gray_prev = cv2.cvtColor(image_prev_turn, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(image_curr_turn, cv2.COLOR_BGR2GRAY)
    
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

    # Step 2: Draw matches (for visualization)
    image_matches = cv2.drawMatches(gray_prev, kp1, gray_curr, kp2, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the matched keypoints
    #plt.imshow(image_matches)
    #plt.title('Feature Matches')
    #plt.show()

    # Step 3: Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Step 4: Calculate homography matrix to align the images using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

    # Inverse the homography matrix to warp the 'after' image to the 'before' image
    M_inv = np.linalg.inv(M)

    # Step 5: Warp the 'after' image to align with the 'before' image
    height, width = gray_prev.shape
    aligned_after = cv2.warpPerspective(gray_curr, M_inv, (width, height))

    # Step 6: Calculate the absolute difference between the images
    diff_image = cv2.absdiff(gray_prev, aligned_after)

    # Step 7: Threshold the difference image to highlight changes
    _, thresh_diff = cv2.threshold(diff_image, 50, 255, cv2.THRESH_BINARY)

    # Step 8: Find contours in the thresholded difference image
    contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    # Step 9: Draw bounding boxes and annotate with numbers
    valid_holes = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate the perimeter and area for circularity check
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
            
        if perimeter == 0:
            continue
        
        aspect_ratio = w / h if h != 0 else 0
        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter ** 2)
            
        # Condition 1: Aspect ratio should not be too close to 1 (avoid squares and circles)
        # Condition 2: Circularity should not be too close to 1 (avoid circles)
        #if 0.1 < aspect_ratio < 2:  # aspect ratio threshold for elongated shapes
        #    if 0.1 < circularity < 2:
        if True:
            if 6 < w < 45 and 6 < h < 45:
                if 30 < area<1000: #50 - 500 pixels la range cua cac lo dan tu nho den to (bia so 4) neu bia so 8 thi co the nho hon
                    print(area, x, y, w, h)
                    filtered_contours.append(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    if not is_hole_already_exist(x, y, w, h):
                        valid_holes.append((x, y, w, h))
                        #cv2.putText(image_curr_turn, str(f"{area}"), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        result = {"name": f"{lane}-{turn}",
                                "lane": lane,
                                "turn": turn,
                                "holes": valid_holes,
                                "result_text": ""
                                }
                    
                        results.append(result)
    result_text = calculate_score(lane, turn, target)
    for i, result in enumerate(results):
        if (result["turn"] == turn):
            results[i]["result_text"] = result_text
            print(result["result_text"])
            for (x, y, w, h) in result['holes']:
                #area = w*h
                #perimeter = 2*(w+h)
                #print(f"{turn}-{x,y}-{w,h}-{w/h}-{4 * np.pi * area / (perimeter ** 2)}-{area} {perimeter}")
                cv2.rectangle(image_curr_turn, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(image_curr_turn, str(f"{turn,x,y}"), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    # Step 10: Show the images with bounding boxes and numbers
    (a, b, h, k, angle) = get_center_ellipse_parameters(image_curr_turn)
    draw_debug_elipse(image_curr_turn,a, b, h, k, angle)
    cv2.imshow('Bullet Holes Detected', image_curr_turn)
    save_image(image_curr_turn, lane, turn, target)
    #cv2.imshow('Aligned Before Image', aligned_after)
    #cv2.imshow('Difference Image', diff_image)
    #cv2.imshow('Thresholded Difference (Bullet Hole)', thresh_diff)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image_curr_turn, result_text
if __name__=="__main__":
    compare_and_detect(1, 1, "BiaSo4")