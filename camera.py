import cv2
import threading
import os


# khi gan camera vao may tinh, cho phep user init camera va add vao array cameras
class Camera:
    def __init__(self, lane=None, target=None, camera_id = 0):
        self.cap = None
        self.lane = lane  # Lane the camera is focused on
        self.target = target  # Target object or area the camera is aimed at

        # Check if the camera opened successfully
        #if not self.cap.isOpened():
        #    print("Error: Could not open the camera.")
        #    exit()

    def capture_image(self, turn):
        # Capture a single frame from the camera
        # Initialize the camera object
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Open the default camera
        # Wait for the webcam to initialize
        while not self.cap.isOpened():
            print("Waiting for the webcam to initialize...")
            cv2.waitKey(100)  # Wait for 100 ms before checking again

        print("Webcam initialized successfully!")
        # Set resolution to 1080p
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Verify if the resolution is set correctly
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Resolution: {width}x{height}")
        cv2.waitKey(1000)
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame")
        else:
            if frame is None or frame.size == 0:
                print("Error: Captured frame empty")
            else:
                self.save_image(frame, turn)
                
            return frame
        
    def show_image(self, frame):
        # Display the captured frame in a window
        cv2.imshow("Captured Image", frame)

    def save_image(self, frame, turn):
        # if folder lane not exist, create new
        # Save the captured frame as a file
        path = f"./Images/Lane{self.lane}/"
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(f"{path}{self.target}-{self.lane}-{turn}.jpg", frame)

    def set_lane(self, lane):
        # Set the lane property
        self.lane = lane

    def set_target(self, target):
        # Set the target property
        self.target = target

    def get_lane(self):
        # Get the lane property
        return self.lane

    def get_target(self):
        # Get the target property
        return self.target

    def release(self):
        # Release the camera and close any OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

def parallel_capture(cameras, turn):
    
    threads = []

    # Create threads for each camera
    for camera in cameras:
        t = threading.Thread(target=camera.capture_image, args=(turn,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    cv2.destroyAllWindows()

cameras = []
camera_1 = Camera(lane=1, target="BiaSo8", camera_id=1)
cameras.append(camera_1)

# Example usage:
if __name__ == "__main__":
    # Create a Camera object with initial lane and target
    camera_0 = Camera(lane=1, target="BiaSo8", camera_id=0)
    camera_1 = Camera(lane=1, target="BiaSo4", camera_id=1)
    cameras = []
    cameras.append(camera_0)
    cameras.append(camera_1)

    #parallel_capture(cameras, 1)
    camera_1.capture_image(1)
    # Display the current lane and target

