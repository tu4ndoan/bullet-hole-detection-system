import cv2
import numpy as np

# Load the image
image = cv2.imread('./Images/Lane1/BiaSo8-1-0.jpg')

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
    if len(contour) >= 5:  # Fit an ellipse requires at least 5 points
        # Fit an ellipse to the contour
        ellipse = cv2.fitEllipse(contour)

        # Extract ellipse parameters (center, axes, angle)
        center, axes, angle = ellipse

        # Compute the aspect ratio (major axis / minor axis) to check if it's "perfect"
        aspect_ratio = max(axes) / min(axes)

        # If the aspect ratio is close to 1, it is more likely a perfect circle (or close ellipse)
        if aspect_ratio > 0.8 and aspect_ratio < 1.2:  # Allow for slight variation in a perfect circle
            print(f"Perfect ellipse found at: Center: {center}, Axes: {axes}, Angle: {angle}")

            # Draw the ellipse on the image
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)  # Green ellipse
            cv2.circle(image, (int(center[0]), int(center[1])), 3, (0, 0, 255), 5)  # Red dot at center

# Display the image with detected ellipses
cv2.imshow("Detected Ellipses", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
