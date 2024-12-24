import cv2

# Load the image in grayscale
image = cv2.imread('./Images/Lane1/test-1-2.jpg', cv2.IMREAD_GRAYSCALE)

# Apply adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(
    image, 
    255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 
    11,  # Block size (local region size)
    2    # Constant subtracted from mean or weighted mean
)

# Display the result
cv2.imshow("Adaptive Thresholded Image", adaptive_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
