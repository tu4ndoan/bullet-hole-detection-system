import cv2
import threading

cameras = []
# khi gan camera vao may tinh, cho phep user init camera va add vao array cameras
class Camera:
    def __init__(self, lane=None, target=None, camera_id = 0):
        # Initialize the camera object
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # Open the default camera
        self.lane = lane  # Lane the camera is focused on
        self.target = target  # Target object or area the camera is aimed at

        # Check if the camera opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open the camera.")
            exit()

    def capture_image(self, turn):
        # Capture a single frame from the camera
        ret, frame = self.cap.read()
        if ret:
            self.save_image(frame, turn)
            return frame
        else:
            print("Error: Failed to capture image.")
            return None

    def show_image(self, frame):
        # Display the captured frame in a window
        cv2.imshow("Captured Image", frame)

    def save_image(self, frame, turn):
        # Save the captured frame as a file
        path = f"./Images/Lane{self.lane}/"
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

# Example usage:
if __name__ == "__main__":
    # Create a Camera object with initial lane and target
    camera_0 = Camera(lane=1, target="BiaSo8", camera_id=0)
    camera_1 = Camera(lane=1, target="BiaSo4", camera_id=1)
    cameras = []
    cameras.append(camera_0)
    cameras.append(camera_1)

    parallel_capture(cameras, 1)
    # Display the current lane and target

