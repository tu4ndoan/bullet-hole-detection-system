import cv2
import numpy as np


def straighthen_img():
    # Load the image
    image = cv2.imread('./Images/Lane1/camera_1.jpg')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection (Canny)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on area (largest contour is usually the target)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find the bounding box for the largest contour (the target)
    target_contour = contours[0]
    rect = cv2.minAreaRect(target_contour)  # Get the rotated bounding box
    box = cv2.boxPoints(rect)  # Get the four points of the box
    box = np.int32(box)

    # Compute the angle of rotation of the bounding box
    angle = rect[2]

    # Correct the angle (to make sure we are rotating correctly)
    if angle < -45:
        angle = 90 + angle

    # Get the rotation matrix
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation to straighten the target
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Crop the image to the region of interest (the target area)
    rotated_box = cv2.polylines(rotated_image, [box], isClosed=True, color=(0, 255, 0), thickness=2)

    # Show the result
    cv2.imshow('Straightened Target Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

straighthen_img()
# Load the reference image (a stable target image) and the image to stabilize
reference_image = cv2.imread('./Images/Lane1/test-1-1.jpg')
image_to_stabilize = cv2.imread('./Images/Lane1/test-1-0.jpg')



"""
# Convert images to grayscale
gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.cvtColor(image_to_stabilize, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
gray_reference = cv2.GaussianBlur(gray_reference, (5, 5), 0)
gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Detect ORB keypoints and descriptors
orb = cv2.ORB_create(nfeatures=1500)  # Increase the number of keypoints
kp1, des1 = orb.detectAndCompute(gray_reference, None)
kp2, des2 = orb.detectAndCompute(gray_image, None)

# Use BFMatcher to match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Find matches and apply Lowe's ratio test
matches = bf.match(des1, des2)
good_matches = [m for m, n in zip(matches[:-1], matches[1:]) if m.distance < 0.75 * n.distance]

# Extract matched points
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute the homography matrix with RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)  # Adjust threshold

# Apply the homography to stabilize the current image
height, width, channels = reference_image.shape
stabilized_image = cv2.warpPerspective(image_to_stabilize, H, (width, height))

# Show the stabilized image
cv2.imshow('Stabilized Target Image', stabilized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""