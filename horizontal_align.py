import cv2
import numpy as np

# Load the image
image = cv2.imread('./Images/Lane1/test.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny
edges = cv2.Canny(gray_blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area (largest first)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Get the largest contour (the shooting target) #this is not correct
target_contour = contours[0]

# Get the rotated bounding box of the target using minAreaRect
rect = cv2.minAreaRect(target_contour)
box = cv2.boxPoints(rect)  # Get the four points of the bounding box
box = np.int32(box)

# Compute the angle of the rotated bounding box
angle = rect[2]

# If the angle is negative, adjust it to be in the correct range
if angle < -45:
    angle = 90 + angle

# Get the rotation matrix
height, width = image.shape[:2]
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

# Apply the rotation to align the target horizontally
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Optionally, crop the image to the rotated target's bounding box (if desired)
# rotated_box = cv2.polylines(rotated_image, [box], isClosed=True, color=(0, 255, 0), thickness=2)

# Show the rotated (horizontally aligned) image
cv2.imshow('Horizontally Aligned Target', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
