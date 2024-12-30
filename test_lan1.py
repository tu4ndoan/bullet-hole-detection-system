import cv2
import camera
import object_detection
import image_processing
import numpy as np

# Initialize the GaussianBlurdow
cv2.namedWindow("GaussianBlur")

# Create trackbars to adjust kernel size
cv2.createTrackbar("blur", "GaussianBlur", 1, 30, lambda x: None)  # Kernel size (1-30)
cv2.createTrackbar("b", "GaussianBlur", 1, 255, lambda x: None)
cv2.createTrackbar("a", "GaussianBlur", 1, 255, lambda x: None)
cv2.createTrackbar("o", "GaussianBlur", 1, 255, lambda x: None)
cv2.createTrackbar("block_size", "GaussianBlur", 3, 255, lambda x: None)
cv2.createTrackbar("lower_edge_val", "GaussianBlur", 1, 255, lambda x: None)
cv2.createTrackbar("upper_edge_val", "GaussianBlur", 1, 255, lambda x: None)


# Load an image
cam_1 = camera.Camera(1,"test", 1)
image = cam_1.capture_image(1)
cropped_target, target_masked, target_mask = object_detection.detect_target(image)
zoomed = object_detection.zoom_in(cropped_target, 1)
while True:
    # Get the current value of the trackbar (kernel size)
    blur = cv2.getTrackbarPos("blur", "GaussianBlur")
    thresh_b_val = cv2.getTrackbarPos("b", "GaussianBlur")
    thresh_a_val = cv2.getTrackbarPos("a", "GaussianBlur")
    thresh_o_val = cv2.getTrackbarPos("o", "GaussianBlur")
    block_size = cv2.getTrackbarPos("block_size", "GaussianBlur")
    lower_edge_val = cv2.getTrackbarPos("lower_edge_val", "GaussianBlur")
    upper_edge_val = cv2.getTrackbarPos("lower_edge_val", "GaussianBlur")

# Ensure kernel size is odd
    if blur % 2 == 0:
        blur += 1
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 1:
        block_size += 1
    gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
    
    # Histogram Equalization to improve contrast
    gray_equalized = cv2.equalizeHist(gray)

    # Apply Gaussian Blur to smooth lighting variations
    gray_blurred = cv2.GaussianBlur(gray_equalized, (blur, blur), 0)
    
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
    _, thresh_binary = cv2.threshold(thresh_adaptive, thresh_b_val, 255, cv2.THRESH_BINARY)
    #_, thresh_b = cv2.threshold(thresh_binary, thresh_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Edge detection with dynamic thresholds
    median_intensity = np.median(gray_blurred)
    lower_thresh = max(0, median_intensity - lower_edge_val)
    upper_thresh = min(255, median_intensity + upper_edge_val)
    edges = cv2.Canny(thresh_binary, threshold1=lower_thresh, threshold2=upper_thresh)

    
    # Show the blurred image
    cv2.imshow("GaussianBlur", edges)

    # Break the loop if the user presses the 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cv2.destroyAllGaussianBlurdows()
