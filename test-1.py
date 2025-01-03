import cv2
import numpy as np

def detect_concentric_ellipses(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Use edge detection (Canny)
    edges = cv2.Canny(adaptive_thresh, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store the detected ellipses
    ellipses = []
    
    # Iterate through the contours and fit ellipses
    for contour in contours:
        if len(contour) >= 5:  # At least 5 points required for fitting an ellipse
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
    
    # If no ellipses detected
    if len(ellipses) == 0:
        print("No ellipses found.")
        return
    
    # Find the center of the first ellipse (assuming they are concentric)
    center_x, center_y = ellipses[0][0]
    
    # Draw ellipses and their centers
    for ellipse in ellipses:
        center, axes, angle = ellipse
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)  # Green ellipses
        cv2.circle(image, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)  # Red centers
    
    # Show the result
    cv2.imshow('Detected Ellipses', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_concentric_ellipses('./HinhAnh/DaiBan1/BiaSo4-1-2.jpg')
