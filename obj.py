import cv2
import numpy as np

# Step 1: Load the image
image = cv2.imread('./Images/Lane1/BiaTest-1-1.jpg')

# Step 2: Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Gaussian Blur to reduce noise and improve thresholding
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Step 4: Apply adaptive thresholding to get a binary image
threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 5)

# Step 5: Find contours in the thresholded image
contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Draw contours on the original image to visualize the detected objects
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2

# Step 7: Display the final image with contours
cv2.imshow('Detected Objects', threshold_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
