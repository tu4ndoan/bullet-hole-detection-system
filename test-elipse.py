import cv2
import numpy as np

def detect_bullet_holes(image_path, target_contours=None, camera_matrix=None, dist_coeffs=None):
    """
    Detect bullet holes in an image, accounting for perspective distortion and camera angle.
    
    Args:
        image_path (str): Path to the input image.
        target_contours (list of contours, optional): Known contours for perspective correction.
        camera_matrix (np.array, optional): Camera calibration matrix for homography correction.
        dist_coeffs (np.array, optional): Distortion coefficients for lens distortion correction.
        
    Returns:
        List of detected bullet holes in the form of ellipses (center, axes, angle).
    """
    # Step 1: Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Undistort the image if a camera calibration is provided
    if camera_matrix is not None and dist_coeffs is not None:
        image = cv2.undistort(image, camera_matrix, dist_coeffs)

    # Step 3: Preprocess image (Thresholding and Edge Detection)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 100, 200)

    # Step 4: Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Detect bullet holes (circles/ellipses)
    detected_bullet_holes = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter out small contours
            # Step 6: Fit ellipse to contour (handles perspective distortion)
            if len(contour) >= 5:  # At least 5 points required to fit an ellipse
                ellipse = cv2.fitEllipse(contour)
                detected_bullet_holes.append(ellipse)

    # Step 7: Optionally, apply perspective correction (homography) if target contours are known
    if target_contours is not None:
        # Compute homography matrix if target contours are provided
        homography_matrix, _ = cv2.findHomography(np.array(target_contours), np.array(detected_bullet_holes))
        corrected_image = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))
        return detected_bullet_holes, corrected_image

    return detected_bullet_holes

def draw_detected_bullet_holes(image, bullet_holes):
    """
    Draw the detected bullet holes (ellipses) on the image.
    
    Args:
        image (np.array): Input image.
        bullet_holes (list): List of detected bullet holes (ellipses).
    
    Returns:
        np.array: Image with bullet holes drawn.
    """
    for ellipse in bullet_holes:
        center, axes, angle = ellipse
        color = (0, 255, 0)  # Green color
        cv2.ellipse(image, center, (int(axes[0] / 2), int(axes[1] / 2)), angle, 0, 360, color, 2)
    return image

# Example usage
image_path = './Images/Lane1/BiaTest-1-1.jpg'  # Replace with your image file path
camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])  # Example camera matrix
dist_coeffs = np.array([0, 0, 0, 0])  # Example distortion coefficients

# Target contours (could be your predefined target surface or previous detection)
target_contours = [
    [(100, 100), (200, 100), (200, 200), (100, 200)]  # Example, replace with actual data
]

bullet_holes = detect_bullet_holes(image_path, target_contours=target_contours, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# Draw the detected bullet holes on the original image
image = cv2.imread(image_path)
image_with_bullet_holes = draw_detected_bullet_holes(image, bullet_holes)

# Display the result
cv2.imshow('Detected Bullet Holes', image_with_bullet_holes)
cv2.waitKey(0)
cv2.destroyAllWindows()
