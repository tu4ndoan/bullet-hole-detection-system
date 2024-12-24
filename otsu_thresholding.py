import cv2

# Load the image in grayscale
image = cv2.imread('./Images/Lane1/test-1-2.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Otsu's thresholding
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the result
cv2.imshow("Otsu's Thresholded Image", otsu_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
