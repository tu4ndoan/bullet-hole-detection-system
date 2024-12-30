import image_processing
import camera
import object_detection
import cv2
import numpy as np

cam_1 = camera.Camera(1,"test", 1)
image = cam_1.capture_image(1)
results = []
def circularity_check(image, x, y, r, min_contrast=0.1, max_circularity=1.2):
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
    #cv2.circle(mask, (r, r), r, 255, -1)  # Full circle mask
    
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
        if circularity <= max_circularity and contrast >= min_contrast:
            return True
    
    return False

def detect_bullet_hole_test(image, turn, lane, target, gray_blur_value, block_size, thresh_a_val ,thresh_value, edge_thresh_1, edge_thresh_2,  min_dist, param1, param2, min_rad, max_rad):
    # Pre-processing
    cropped_target, target_masked, target_mask = object_detection.detect_target(image)
    zoomed = object_detection.zoom_in(cropped_target, 1)
    
    gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = cv2.equalizeHist(gray)
    
    
    thresh_adaptive = cv2.adaptiveThreshold(
        gray_eq, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    

    #_, thresh_binary = cv2.threshold(thresh_adaptive, thresh_value, 255, cv2.THRESH_BINARY)
   # _, thresh_b = cv2.threshold(thresh_adaptive, thresh_value, 255, cv2.THRESH_BINARY)
    gray_blurred = cv2.GaussianBlur(thresh_adaptive, (9, 9), 2)
    median_intensity = np.median(gray_blurred)
    lower_thresh = max(0, median_intensity - edge_thresh_1)
    upper_thresh = min(255, median_intensity + edge_thresh_2)
    edges = cv2.Canny(gray_blurred, threshold1=100, threshold2=200)

    circles = cv2.HoughCircles(
        edges, 
        cv2.HOUGH_GRADIENT, 
        dp=0.8, 
        minDist=30, 
        param1=100, 
        param2=30, 
        minRadius=5, 
        maxRadius=15
    )
    
    holes = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        valid_circles = []

        for circle in circles:
            x, y, r = circle
            print(f"Detected circle at ({x}, {y}) with radius {r}")
            
            # Use the robust circularity and contrast check
            if True:
                valid_circles.append(circle)
                holes.append((x, y, r))

        result = {
            "name": f"{lane}-{turn}",
            "lane": lane,
            "turn": turn,
            "holes": holes
        }
        results.append(result)

        for result in results:
            print(f"Turn {result['turn']} - Number of bullet holes: {len(result['holes'])}")
            for (x, y, r) in result['holes']:
                draw_debug(zoomed, x, y, r, result["turn"])

    cv2.imshow('Processed Image', zoomed)
    cv2.imshow("Edge Detection", edges)

    #calculate_score(lane, turn)
    #save_image(zoomed, lane, turn, target)  

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_debug(image, x, y, r, turn):
    """
    Draws debug circles on the image to visualize detected bullet holes.
    """
    cv2.circle(image, (x, y), r, (0, 255, 0), 4)
    cv2.putText(image, f"Turn {turn}", (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


#image_processing.detect_bullet_hole(img, 1, 1, "test", 5, 200, 150, 150, 10, 150, 16, 2, 20)
# 5, 200, 150, 150, 50, 150, 16, 2, 20 la cac tham so cua ham detect_bullet_hole
# đã test với điều kiện ngược nắng, góc nghiêng 45 độ, khoảng cách 70cm, phòng không bật đèn, buổi trưa 3h30, ngược nắng
# kết quả: phát hiện đầy đủ lỗ đạn, không bị gồ ghề, mặt phẳng edge OK

# case 2: buổi tối 19h, ánh sáng trực tiếp chiếu vào bia 
# phương án: thay đổi tham số
# thêm adaptive thresh, histogram
# ánh sáng trực tiếp chiếu vào bia thì bị hiện các chi tiết -> nhận rất nhiều lỗ sai -> TODO: improve threshold 
image_processing.detect_bullet_hole_test(image, 1, 1, "test", 5, 11, 5, 1, 150, 150, 10, 150, 10, 5, 15)
#detect_bullet_hole_test(img, 1, 1, "test", 15, 11, 5, 200, 150, 150, 20, 150, 20, 16, 20)# với ánh sáng vừa, ngược sáng 1 chút, không nắng 8h sáng thì nhận đc đủ lỗ, nhận nhầm 1 số chi tiết trên bia như số 8 9, súng,...
# note 30/12: đã áp dụng trunc_thresh, kết quả khả quan hơn khoản 80%
# áp dụng link edge
# CLAHE
# TODO: review kết quả, các chi tiết phần đầu (bị chiếu sáng mạnh) đã loại bỏ được, các chi tiết dưới chưa được, 1 số nếp gấp trên bia ảnh hưởng kết quả
# TODO: tìm hiểu về trunc, clahe, link edge
# thử cách của anh Thạch, so sảnh hình ảnh loạt trước sau, match các key feature để phân biệt các điểm khác