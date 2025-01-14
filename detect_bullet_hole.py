import image_processing
import cv2
import tkinter as tk
import numpy as np
import math
import matplotlib.pyplot as plt
from tkinter import ttk
import camera

b_debug = False # if True, show debug images, edge, thresh, matches, remove bg, draw debug ellipses
results = []

def load_target_params(target):
    global thresh_value, min_h_w, max_h_w, min_bullet_hole_area, max_bullet_hole_area, min_hole_circularity, min_hole_ratio, max_hole_ratio, hole_to_hole_distance
    global min_ratio, max_ratio, min_ellipse_area, max_ellipse_area, min_angle, max_angle, delta_a, delta_k
    if target == "BiaSo4":
        print("Đang tải các tham số của bia số 4")
        # bullet detection params
        thresh_value = 50

        min_h_w = 1 #14
        max_h_w = 30 #20

        min_bullet_hole_area = 20
        max_bullet_hole_area = 100
        min_hole_circularity = 0.4
        min_hole_ratio = 1
        max_hole_ratio = 2
        hole_to_hole_distance = 50

        # ellipse detection params
        min_ratio = 1
        max_ratio = 1.2

        min_ellipse_area = 300000
        max_ellipse_area = 400000

        min_angle = 80
        max_angle = 100

        delta_k = 0.5
        delta_a = 0.2

    elif target == "BiaSo7":
        print("Đang tải các tham số của bia số 7")

        # bullet detection params
        thresh_value = 50

        min_h_w = 1 #8
        max_h_w = 15 #15

        min_bullet_hole_area = 5
        max_bullet_hole_area = 50
        min_hole_circularity = 0.3
        min_hole_ratio = 1
        max_hole_ratio = 2
        hole_to_hole_distance = 35

        # ellipse detection params
        min_ratio = 1
        max_ratio = 2

        min_ellipse_area = 40000
        max_ellipse_area = 50000

        min_angle = 170
        max_angle = 190

        delta_k = 0.5
        delta_a = 0.5

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
    print("Đang kiểm tra các lỗ đạn này có trùng với lỗ đạn trước không...")
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
                        print("Phát hiện lỗ đạn trùng nhau")
                        return True
                    else:
                        return True
    return False

def circularity_check(image,x, y, r):
    print("Đang kiểm tra độ tròn của lỗ đạn")
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

def get_center_ellipse_parameters(image, target):
    processed = image_processing.preprocess_image(image)
    
    adaptive_thresh = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    _,thresh = cv2.threshold(adaptive_thresh, 100, 255, cv2.THRESH_BINARY)
    # Use edge detection (Canny)
    edges = cv2.Canny(thresh, 100, 150)
   
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

            if a * b == 0:
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
                if b_debug:
                    cv2.ellipse(image, (int(h), int(k)), (int(a), int(b)), angle, 0, 360, (0, 0, 255), 2)  # Green color, thickness 2
            if b_debug:
                cv2.ellipse(image, (int(h), int(k)), (int(a), int(b)), angle, 0, 360, (0, 0, 255), 2)  # Green color, thickness 2
                print(f"Found ellipse: Area={area}, Aspect Ratio={aspect_ratio}, Angle={angle}, Position=({h}, {k} Axes=({a//4}, {b//4})")
    

    # Get image dimensions
    height, width = image.shape[:2]

    # Default values for ellipse parameters in case no ellipse is found
    center_h, center_k = int(width // 2), int(height // 2)

    # default ellipse nếu không xác định được ellipse
    if target == "BiaSo4":
        a, b, h, k, angle = 74, 84, center_h, center_k, 90 #default if no ellipse found
    elif target == "BiaSo7":
        a, b, h, k, angle = 65, 45, center_h, int(center_k//2), 90 #default if no ellipse found

    if center_ellipse:
        (h, k) = center_ellipse[0]  # Center
        (a, b) = center_ellipse[1]  # Axes
        angle = center_ellipse[2]   # Angle
        
        print(f"Đã xác định được ellipse {a, b, h, k, angle}")

    return int(a), int(b), int(h), int(k), int(angle)

# draw
def draw_debug_elipse(image, a, b, h, k, angle, target):
    
    # rotation angle in degrees
    start_angle = 0  # starting angle of the arc
    end_angle = 360  # ending angle of the arc (full ellipse)
    if target == "BiaSo4": # vì bia số 4 detect được vòng ellipse thứ 4
        a = a // 4  # Scale down major axis for your use case
        b = b // 4  # Scale down minor axis for your use case
        for i in range(1,6): # bia so 4
        # Draw the ellipse on the image
            if i == 1:
                cv2.circle(image, (int(h),int(k-5)), 1, (0,0,255), 2)
            k += 2*i
            a += 0.5*i
            b += 0.5*i

            cv2.ellipse(image, (int(h),int(k)), (int(a)*i,int(b)*i), angle, start_angle, end_angle, (0, 255, 0), 2)
    elif target == "BiaSo7": # vì bia số 7 detect được vòng ellipse thứ 2
        a = a // 2  # Scale down major axis for your use case
        b = b // 2  # Scale down minor axis for your use case
        for i in range(1,11):
            if i == 1:
                cv2.circle(image, (int(h),int(k-5)), 1, (0,0,255), 2)
            k += 1*i
            b += 0.2*i

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
    if target == "BiaSo4":
        a = a//4
        b = b//4
        for j in range(1, 11):
            k += 2*j
            a += 0.5*j
            b += 0.5*j
    elif target == "BiaSo7":
        a = a//2
        b = b//2
        for j in range(1, 11):
            k += 1*j
            b += 0.2*i
    
    for (x, y, _, _) in holes:
        
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
    print(f"Đang tải tham số cho mục tiêu {target}")
    load_target_params(target)
    try:
        print("Đang tải lên các hình ảnh loạt trước...")
        image_prev_turn = image_processing.load_image(lane, turn-1, target)
        print("Đang tải lên các hình ảnh loạt này...")
        image_curr_turn = image_processing.load_image(lane, turn, target)
        print("Đang xử lý hình ảnh...")
        gray_prev = image_processing.preprocess_image(image_prev_turn)
        gray_curr = image_processing.preprocess_image(image_curr_turn)
        print("Đang khử độ nhiễu của ảnh")
        gray_prev = cv2.fastNlMeansDenoising(gray_prev, None, 15, 7, 21) # denoise 15 là đẹp
        gray_curr = cv2.fastNlMeansDenoising(gray_curr, None, 15, 7, 21)
        print("Xử lý hình ảnh xong.")
    except Exception as e:
        print(e)
        return None, None
    
    # Step 1: Feature detection and matching (ORB in this case)
    orb = cv2.ORB_create(nfeatures=1000)
    

    # Detect keypoints and descriptors
    print("Đang tìm các điểm then chốt của ảnh...")
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
    print("Đang so sánh hình ảnh loạt trước và sau...")
    diff_image = cv2.absdiff(aligned_before, gray_curr)

    # Step 7: Threshold the difference image to highlight changes
    _, thresh_diff = cv2.threshold(diff_image, 100, 255, cv2.THRESH_BINARY)
    # Step 8: Find contours in the thresholded difference image
    contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Step 9: Draw bounding boxes and annotate with numbers
    valid_holes = []
    for i, contour in enumerate(contours):
        if not len(contour) >= 5:
            continue
        ellipse = cv2.fitEllipse(contour)
        (a, b) = ellipse[1]  # Major and minor axes
        # Check if any parameter is NaN or invalid
        if np.isnan(a) or np.isnan(b):
            continue
        # Calculate the area of the ellipse
        area = np.pi * a * b

        if area == 0 or a * b == 0:
            continue
        # Calculate aspect ratio
        ellipse_aspect_ratio = max(a, b)/min(a, b)

        x, y, w, h = cv2.boundingRect(contour)
        if w*h == 0:
            continue

        aspect_ratio = max(w,h)/min(w,h)

        # filtering holes
        if is_hole_already_exist(lane, target, x,y,w,h):
            continue
        

        # continue filtering holes
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        contour_area = cv2.contourArea(contour)
        if contour_area < min_bullet_hole_area or contour_area > max_bullet_hole_area:
            continue

        circularity = 4 * np.pi * contour_area / (perimeter ** 2)
        if circularity < min_hole_circularity:
            continue

        if min_hole_ratio > ellipse_aspect_ratio or ellipse_aspect_ratio > max_hole_ratio:
            continue
        
        print(f"Đã phát hiện lỗ đạn: tọa độ x:{x} - y:{y} - w:{w} - h:{h} - contour_area:{contour_area:.2f}- perimeter:{perimeter:.2f} - aspect_ratio:{aspect_ratio:.2f} - circ:{circularity:.2f} - aspect_ratio1:{ellipse_aspect_ratio:.2f}")
        # pass all the check? then append to valid_holes
        valid_holes.append((x, y, w, h))
        cv2.rectangle(image_curr_turn, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(image_curr_turn, str(f"{turn}"), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # construct result
    total_score, result_text = 0, "Bia an toan"
    print("Đang tính điểm...")
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
        
    print("Đang lưu kết quả...")
    image_processing.save_image(image_curr_turn, lane, turn, target)

    # Step 10: Show the images with bounding boxes and numbers (debug only)
    if b_debug: # if package change to b_debug
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
    #cam = camera.Camera(1,"BiaSo4", 1)
    #cam2 = camera.Camera(1,"BiaSo7", 2)
    
    #img = cam.capture_image(1)
    #img2 = cam2.capture_image(1)

    #compare_and_detect(1, 1, "BiaSo4")
    compare_and_detect(1, 3, "BiaSo7")

