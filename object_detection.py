import cv2
import numpy as np
import camera

def preprocess_image(image):
    """
    Preprocess the image to handle lighting variations (strong lighting, overexposure).
    
    Args:
        image (np.array): The input image.
        
    Returns:
        (np.array): The preprocessed image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the image to adjust for lighting conditions
    # Normalize pixel values to 0-255 range
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)

    return gray

def detect_target(image):
    """
    Detects an object (e.g., a shooting target) in the image using contour detection.
    
    Args:
        image (np.array): The input image.
        
    Returns:
        (np.array): The cropped region around the detected target.
    """
    # Step 1: Preprocess the image to handle lighting and contrast
    gray = preprocess_image(image)
    
    # Step 2: Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0) # 5 5
    
   #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to the image
    #clahe_image = clahe.apply(gray_blurred)
    # Adaptive thresholding to account for varying lighting conditions
    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
    ret, otsu_thresh = cv2.threshold(thresh_adaptive, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Edge detection with dynamic thresholds
    median_intensity = np.median(blurred)
    lower_thresh = max(0, median_intensity - 150)
    upper_thresh = min(255, median_intensity + 150)
    edges = cv2.Canny(otsu_thresh, threshold1=lower_thresh, threshold2=upper_thresh) # 150 150

    kernel = np.ones((3, 3), np.uint8)  # A 3x3 kernel for dilation and erosion
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)  # Dilate to join edges
    linked_edges = cv2.erode(dilated_edges, kernel, iterations=1)  # Erode to reduce noise
    
    # Step 4: Find contours in the edge-detected image
    contours, _ = cv2.findContours(linked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None
    
    # Step 5: Find the largest contour (assuming it's the target)
    target_contour = max(contours, key=cv2.contourArea)

    # Step 6: Get the bounding box for the contour
    x, y, w, h = cv2.boundingRect(target_contour)

    # Step 7: Create a mask for the detected target (use the bounding box)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [target_contour], -1, (255, 255, 255), -1)

    # Step 8: Apply the mask to the image (keep the target, remove background)
    target = cv2.bitwise_and(image, mask)

    # Step 9: Crop the region around the detected target
    cropped_image = image[y:y+h, x:x+w]  # Crop using the bounding box coordinates

    return cropped_image, target, mask

def zoom_in(image, zoom_factor=1.5):
    """
    Zooms in on the center of the image by the given zoom factor.
    
    Args:
        image (np.array): The input image.
        zoom_factor (float): Factor by which to zoom in. (e.g., 1.5 means 50% zoom-in).
        
    Returns:
        np.array: Zoomed-in image.
    """
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Calculate the new dimensions for zoom
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    
    # Resize the image to the new dimensions
    zoomed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Crop the center region of the zoomed image
    x_start = (new_width - width) // 2
    y_start = (new_height - height) // 2
    zoomed_in = zoomed_image[y_start:y_start+height, x_start:x_start+width]
    
    return zoomed_in

def remove_background(image):
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
        cv2.imshow("remove bg", background_removed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return background_removed


if __name__ == "__main__":
    # Load the input image
    #image_path = './HinhAnh/DaiBan1/BiaSo4-1-10.jpg'  # Replace with the path to your image
    #image = cv2.imread(image_path)
    cam = camera.Camera(1, "BiaTest", 1)
    image = cam.capture_image(121)

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
