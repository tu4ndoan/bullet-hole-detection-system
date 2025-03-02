import cv2
import tkinter as tk
import numpy as np
import os
import camera
from rembg import remove
from PIL import Image
from tkinter import ttk


# image load/save
def load_image(lane, turn, target):
    image = cv2.imread(f'./HinhAnh/DaiBan{lane}/{target}-{lane}-{turn}.jpg', 1)
    return image

def load_result(lane, turn, target):
    result = cv2.imread(f'./HinhAnh/KetQua/DaiBan{lane}/{target}-{lane}-{turn}-marked.jpg')
    return result

def save_image(image, lane, turn, target):
    lane_dir = f"./HinhAnh/DaiBan{lane+1}"
    result_dir = f"./HinhAnh/KetQua/DaiBan{lane+1}"
    if not os.path.exists(lane_dir):
        os.makedirs(lane_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    cv2.imwrite(f"./HinhAnh/KetQua/DaiBan{lane}/{target}-{lane}-{turn}-marked.jpg", image)

# Preprocess the image to handle lighting variations (strong lighting, overexposure).
def preprocess_image(image):
    # test gamma
    print("Đang điều chỉnh gamma")
    image = adjust_gamma(image, 0.8)
    # Convert to grayscale
    print("Đang chuyển về grayscale")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the image to adjust for lighting conditions
    # Normalize pixel values to 0-255 range
    print("Normalizing pixel values to 0-255 range to adjust for lighting conditions")
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply histogram equalization for better contrast
    #gray = cv2.equalizeHist(gray)
    print("Applying CLAHE equalization 3.0 - 8,8 for better contrast")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray =  clahe.apply(gray)
    print("Applying Gaussian blur 3,3 - 0 to reduce noise")
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    
    return gray

def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

    # 2. Unsharp Masking for sharpening
def unsharp_mask(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # 3. Adaptive Thresholding
def adaptive_threshold(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

    # 5. Contrast and Brightness Adjustment
def adjust_contrast_brightness(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def process_image(image, gamma=1.2, alpha=1.5, beta=0):
    # 1. Gamma Correction
    def adjust_gamma(image, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    # 2. Unsharp Masking for sharpening
    def unsharp_mask(image):
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # 3. Adaptive Thresholding
    def adaptive_threshold(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    def apply_clahe(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray_image)

    # 5. Contrast and Brightness Adjustment
    def adjust_contrast_brightness(image, alpha, beta):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Step 1: Apply Gamma Correction
    image = adjust_gamma(image, gamma)

    # Step 2: Apply Contrast and Brightness Adjustment
    image = adjust_contrast_brightness(image, alpha, beta)

    # Step 3: Apply Unsharp Mask for sharpening
    image = unsharp_mask(image)

    # Step 4: Apply CLAHE for enhancing local details
    image_clahe = apply_clahe(image)

    # Step 5: Perform Edge Detection (Canny) for fine details
    edges = cv2.Canny(image, 50, 150)

    # Step 6: Adaptive Thresholding for handling high contrast and overexposed areas
    thresh_image = adaptive_threshold(image)

    # Return processed image (you can return all or choose one of the results depending on use)
    return image

def linked_edges(gray_equalized):
    
    # Apply Gaussian Blur to smooth lighting variations
    gray_blurred = cv2.GaussianBlur(gray_equalized, (5, 5), 0)
    
    # Adaptive thresholding to account for varying lighting conditions
    thresh_adaptive = cv2.adaptiveThreshold(
        gray_blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        5
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    clahe_image = clahe.apply(thresh_adaptive)
    # Further binary thresholding if needed
    #ret, inv_thresh = cv2.threshold(gray_blurred, 150, 255, cv2.THRESH_BINARY_INV)
    #_, thresh = cv2.threshold(gray_blurred, 150, 255, cv2.THRESH_BINARY)
    #ret, trunc_thresh = cv2.threshold(clahe_image, 55, 255, cv2.THRESH_TRUNC)
    _, otsu_thresh = cv2.threshold(clahe_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Edge detection with dynamic thresholds
    median_intensity = np.median(gray_blurred)
    lower_thresh = max(0, median_intensity - 50)
    upper_thresh = min(255, median_intensity + 150)
    edges = cv2.Canny(otsu_thresh, threshold1=lower_thresh, threshold2=upper_thresh)
    
    
    # Apply dilation and erosion to link fragmented edges
    kernel = np.ones((3, 3), np.uint8)  # A 3x3 kernel for dilation and erosion
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)  # Dilate to join edges
    linked_edges = cv2.erode(dilated_edges, kernel, iterations=1)  # Erode to reduce noise
    return linked_edges

def detect_target(image):
    """
    Detects an object (e.g., a shooting target) in the image using contour detection.
    """
    gray = preprocess_image(image)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Thresholding to handle varying lighting conditions
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Otsu Thresholding
    ret, otsu_thresh = cv2.threshold(thresh_adaptive, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Edge Detection
    median_intensity = np.median(blurred)
    lower_thresh = max(0, median_intensity - 150)
    upper_thresh = min(255, median_intensity + 150)
    edges = cv2.Canny(otsu_thresh, threshold1=lower_thresh, threshold2=upper_thresh)
    
    # Morphological Operations: Dilation + Erosion
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    linked_edges = cv2.erode(dilated_edges, kernel, iterations=2)
    
    # Find Contours
    contours, _ = cv2.findContours(linked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
    
    # Filter out small contours
    contours = [contour for contour in contours if cv2.contourArea(contour) > 100]
    if not contours:
        return None, None, None
    
    # Find largest contour
    target_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding box for the largest contour
    x, y, w, h = cv2.boundingRect(target_contour)
    
    # Create mask for target
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [target_contour], -1, (255, 255, 255), -1)
    
    # Apply mask to get target region
    target = cv2.bitwise_and(image, mask)
    
    # Crop the image around the target
    cropped_image = image[y:y+h, x:x+w]
    
    return cropped_image, target, mask

def zoom_in(image, zoom_factor=1.5):
    """
    Zooms in on the center of the image by the given zoom factor.
    """
    height, width = image.shape[:2]
    
    # New dimensions
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    
    # Resize image
    zoomed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Crop the center region
    x_start = (new_width - width) // 2
    y_start = (new_height - height) // 2
    zoomed_in = zoomed_image[y_start:y_start + height, x_start:x_start + width]
    
    return zoomed_in
# 
def remove_background(image):
    background_removed = remove(image)

    return background_removed.copy()



if __name__ == "__main__":
    cam = camera.Camera(1, "BiaSo7", 2)
    image = cam.capture_image(200)

    # Step 1: Detect the target (e.g., shooting target)
    try:
        cropped_target, target_masked, target_mask = detect_target(image)
    except:
        cropped_target = image

    if cropped_target is not None:
        # Step 2: Zoom in on the detected target
        zoomed_target = zoom_in(cropped_target, zoom_factor=1)

        # Step 3: Resize the target mask to match the size of the zoomed-in target
        target_mask_resized = cv2.resize(target_mask, (zoomed_target.shape[1], zoomed_target.shape[0]), interpolation=cv2.INTER_NEAREST)
        target_mask_inverted = cv2.bitwise_not(target_mask_resized)

        # Step 4: Remove background (by using the resized target mask)
        background_removed = cv2.bitwise_and(image, target_mask)

        # Show the results
        cv2.imshow('Zoomed Target', background_removed)
        # cv2.imshow('Background Removed', background_removed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Target not detected.")