import cv2

# Open the camera (0 is typically the default camera index)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Capture a single frame
ret, frame = cap.read()

# If frame is read successfully, show the image
if ret:
    # Display the captured image in a window
    cv2.imshow("Captured Image", frame)
    
    # Save the captured image as a file (optional)
    cv2.imwrite("captured_image.jpg", frame)
    
    # Wait for a key press and close the window
    cv2.waitKey(0)
else:
    print("Error: Failed to capture image.")

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
