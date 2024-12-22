import cv2
import numpy as np


def load_image(lane, turn, target_name):
    """Load original and modified images."""
    #image_1 = cv2.imread(f'./Images/Lane{lane}/{target_name}-{lane}-{turn-1}.jpg')
    image = cv2.imread(f'./Images/Lane{lane}/{target_name}-{lane}-{turn}.jpg')
    return image

def draw_number(image_2, x, y, w, h, turn_num):
    """Draw a number on the image near the detected contour."""
    number = turn_num  # Example number
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)  # Red color for number
    thickness = 2
    posX = x + w // 4
    posY = y + (4 * h) // 5
    cv2.putText(image_2, str(number), (posX, posY), font, font_scale, color, thickness)

def detect_bullet_hole(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and improve detection
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blur, threshold1=50, threshold2=150)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        edges, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20, 
        param1=50, 
        param2=10, 
        minRadius=4, 
        maxRadius=8
    )

    # If circles are detected, draw them on the image
    if circles is not None:
        # Convert to integers (x, y, r)
        circles = np.round(circles[0, :]).astype("int")
        valid_circles = []
        for circle in circles:
            x,y,r = circle
            area = np.pi * (r**2)
            

            # Example criteria:
            # 1. Filter by radius size (valid radius range: 20-80)
            #if r < 5 or r > 7:
            #    continue

            # 2. Filter by position (avoid circles too close to the image boundaries)
            #if x - r < 0 or y - r < 0 or x + r > image.shape[1] or y + r > image.shape[0]:
            #    continue

            # 3. Add valid circle to the list
            valid_circles.append(circle)

        for circle in valid_circles:
            # Draw the circle in the output image
            #cv2.circle(image, (x, y), r, (0, 255, 0), 2)  # Draw the circle
            x, y, w, h = cv2.boundingRect(circle)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)  # Green rectangle
            draw_number(image, x, y, w, h, 1)
            #cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Draw the center
            print(f"{r} - {x} - {y}")
    # Show the result
    cv2.imshow('Detected Circles', image)
    #cv2.imshow('blur Circles', blur)
    #cv2.imshow('edges Circles', edges)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
