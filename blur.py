import cv2
import numpy as np

# Load the image
image = cv2.imread('./Images/Lane1/BiaSo8-1-1.jpg')

# Convert the image to grayscale (if it's colored)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to create a binary mask (foreground vs background)
# Here, we're using a simple thresholding technique to create a mask.
# You can adjust the parameters based on your image or use advanced methods.
_, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Invert the mask to get the background
background_mask = cv2.bitwise_not(mask)

# Blur the background using Gaussian blur
blurred_background = cv2.GaussianBlur(image, (21, 21), 0)

# Mask the foreground and background
foreground = cv2.bitwise_and(image, image, mask=mask)
blurred_background = cv2.bitwise_and(blurred_background, blurred_background, mask=background_mask)

# Combine the foreground and the blurred background
final_image = cv2.add(foreground, blurred_background)

# Show the final image
cv2.imshow('Blurred Background', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
