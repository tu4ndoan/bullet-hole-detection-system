# Create a blank image to display the trackbars
    img = np.zeros((300, 512, 3), dtype=np.uint8)

    # Create a window to hold the trackbars
    cv2.namedWindow('Trackbars')

    # Create trackbars for various parameters
    cv2.createTrackbar('edge_thresh_1', 'Trackbars', 1, 150, nothing)  # Threshold1 for Canny
    cv2.createTrackbar('edge_thresh_2', 'Trackbars', 1, 150, nothing)
    cv2.createTrackbar('param1', 'Trackbars', 1, 150, nothing)  # Threshold1 for Canny
    cv2.createTrackbar('param2', 'Trackbars', 1, 15, nothing)
    cv2.createTrackbar('min_rad', 'Trackbars', 1, 100, nothing)  # Threshold1 for Canny
    cv2.createTrackbar('max_rad', 'Trackbars', 1, 100, nothing)  # Threshold2 for Canny
    cv2.createTrackbar('Blur', 'Trackbars', 5, 20, nothing)
    # Open the video capture
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # Wait for the webcam to initialize
    while not cap.isOpened():
        print("Waiting for the webcam to initialize...")
        cv2.waitKey(100)  # Wait for 100 ms before checking again

    print("Webcam initialized successfully!")
    # Set resolution to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Verify if the resolution is set correctly
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Resolution: {width}x{height}")
    # Capture a frame
    cv2.waitKey(100)
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame")
    else:
        # Check if the frame is empty (black)
        if frame is None or frame.size == 0:
            print("Error: Captured frame is empty")
        else:
            # Save the frame as an image
            cv2.imwrite('captured_frame.jpg', frame)
            print("Frame saved successfully")

    # Capture video frames
    while True:
        ret, frame = cap.read()
        edge_thresh_1 = cv2.getTrackbarPos('edge_thresh_1', 'Trackbars')
        edge_thresh_2 = cv2.getTrackbarPos('edge_thresh_2', 'Trackbars')
        param1 = cv2.getTrackbarPos('param1', 'Trackbars')
        param2 = cv2.getTrackbarPos('param2', 'Trackbars')
        min_rad = cv2.getTrackbarPos('min_rad', 'Trackbars')
        max_rad = cv2.getTrackbarPos('max_rad', 'Trackbars')
        blur_kernel = cv2.getTrackbarPos('Blur', 'Trackbars')
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        #detect bullet hole
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = adaptive_thresholding(gray_blurred)
        #_, otsu_thresh = otsu_thresholding(gray_blurred)
        edges = cv2.Canny(thresh, threshold1=50, threshold2=150)
        
        circles = cv2.HoughCircles(
            edges, 
            cv2.HOUGH_GRADIENT, 
            dp=1.1, 
            minDist=100, 
            param1=150, 
            param2=14, 
            minRadius=5, 
            maxRadius=15
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            valid_circles = []
            holes = []
            # luu ket qua
            for circle in circles:
                x, y, r = circle
                hole = (x,y,r)
                print(x, y, r)
                # check xem hole nay co trung voi loat truoc khong
                if not is_hole_already_exist(x,y,r):
                    valid_circles.append(circle)
                    holes.append(hole)

            result = {"name": f"{1}-{1}",
                    "lane": 1,
                    "turn": 1,
                    "holes": holes
                    }
            results.append(result)

            for result in results:
                #print(f"loat {result["turn"]} ban trung : {len(result["holes"])} phat dan")
                for (x,y,r) in result["holes"]:
                    draw_debug(frame, x,y,r,result["turn"])

        # Display the frame
        cv2.imshow('Video Frame', edges)

        # Break the loop when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()