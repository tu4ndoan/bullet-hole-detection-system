import cv2

# Open the video capture
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set resolution to 1080p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Verify if the resolution is set correctly
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution: {width}x{height}")

# Capture video frames
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame")
        break
    
    #detect bullet hole
    

    # Display the frame
    cv2.imshow('Video Frame', frame)

    # Break the loop when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
