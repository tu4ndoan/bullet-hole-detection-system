import cv2
import numpy as np

# Load the image
image = cv2.imread('./Images/Lane1/BiaSo4-1-0.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

# Apply a binary threshold to separate foreground and background
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Create a mask and apply it to the original image
mask = cv2.bitwise_not(thresh)
result = cv2.bitwise_and(image, image, mask=mask)

# Display the result
cv2.imshow('Original', image)
cv2.imshow('Threshold Mask', thresh)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()