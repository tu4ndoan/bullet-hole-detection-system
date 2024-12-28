import cv2
import camera
import object_detection

# Initialize the GaussianBlurdow
cv2.namedWindow("GaussianBlur")

# Create trackbars to adjust kernel size
cv2.createTrackbar("blur", "GaussianBlur", 1, 30, lambda x: None)  # Kernel size (1-30)
cv2.createTrackbar("b", "GaussianBlur", 1, 255, lambda x: None)
cv2.createTrackbar("a", "GaussianBlur", 1, 255, lambda x: None)
cv2.createTrackbar("o", "GaussianBlur", 1, 255, lambda x: None)
cv2.createTrackbar("block_size", "GaussianBlur", 1, 255, lambda x: None)
cv2.createTrackbar("edges_val", "GaussianBlur", 1, 255, lambda x: None)

# Load an image
zoomed = cv2.imread("./Images/Lane1/test-1-1.jpg")
cam_1 = camera.Camera(1,"test", 1)
image = cam_1.capture_image(1)
cropped_target, target_masked, target_mask = object_detection.detect_target(image)
zoomed = object_detection.zoom_in(cropped_target, 1)
while True:
    # Get the current value of the trackbar (kernel size)
    blur = cv2.getTrackbarPos("blur", "GaussianBlur")
    thresh_b_val = cv2.getTrackbarPos("b", "GaussianBlur")
    thresh_a_val = cv2.getTrackbarPos("a", "GaussianBlur")
    thresh_o_val = cv2.getTrackbarPos("o", "GaussianBlur")
    block_size = cv2.getTrackbarPos("block_size", "GaussianBlur")
    edges_val = cv2.getTrackbarPos("edges_val", "GaussianBlur")
# Ensure kernel size is odd
    if blur % 2 == 0:
        blur += 1
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 1:
        block_size += 1
    gray = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
    _, thresh_binary = cv2.threshold(gray_blurred, thresh_b_val, 255, cv2.THRESH_BINARY) # da test ok voi anh Tuan 9,200,150,150,100,150,11,1,11, van con bi mat phang nhap nho
    #_, thresh_a = adaptive_thresholding(gray_blurred)
    _, thresh_o = cv2.threshold(thresh_binary, thresh_o_val, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(thresh_o ,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, thresh_a_val) # used for various lighting condition
    edges = cv2.Canny(thresh, threshold1=edges_val, threshold2=edges_val)
    
    
    # Show the blurred image
    cv2.imshow("GaussianBlur", edges)

    # Break the loop if the user presses the 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cv2.destroyAllGaussianBlurdows()
