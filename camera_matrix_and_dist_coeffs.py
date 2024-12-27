import cv2
import numpy as np
import glob

# Prepare object points (3D world coordinates for the checkerboard corners)
# Assume a 9x6 checkerboard with 1x1 square size (in any unit, e.g., centimeters)
object_points = np.zeros((6 * 9, 3), np.float32)
object_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points
obj_points = []  # 3D points in world space
img_points = []  # 2D points in image plane

# Load all the images from a folder
images = glob.glob('calibration_images/*.jpg')  # Path to your calibration images

for image_path in images:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the corners of the checkerboard
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        obj_points.append(object_points)  # Append object points (same for all images)
        img_points.append(corners)  # Append image points (2D)

        # Draw and display the corners (optional)
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        cv2.imshow('Checkerboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None)

# Output the camera matrix and distortion coefficients
print("Camera Matrix:")
print(camera_matrix)

print("Distortion Coefficients:")
print(dist_coeffs)

# You can save these parameters for later use
np.savez("calibration_params.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
"""Steps Explained:
Prepare Object Points:

The object_points represent the 3D coordinates of the checkerboard corners in the world space. For a checkerboard with 9x6 squares, we assume the corners are placed at unit distance (1x1) in 3D.
Capture Images:

Use glob to load all the images from a folder.
Find Checkerboard Corners:

For each image, use cv2.findChessboardCorners() to detect the corners of the checkerboard.
Camera Calibration:

cv2.calibrateCamera() computes the camera matrix and distortion coefficients using the object points and image points collected from the multiple images.
Output:

Print the camera matrix and distortion coefficients to the console.
Optionally save them using np.savez() for later use.
How to Use the Camera Matrix and Distortion Coefficients:
After calibration, you can use the camera matrix and distortion coefficients to undistort images and perform 3D reconstruction or other tasks. For example, to undistort an image, you can use:

python
Sao chép mã
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
This will remove the lens distortion from the image using the camera parameters you've obtained.

Summary:
Camera Matrix: Contains intrinsic parameters like focal length and optical center.
Distortion Coefficients: Corrects lens distortion (radial and tangential).
How to find them: Perform camera calibration using a checkerboard pattern and OpenCV functions like cv2.findChessboardCorners() and cv2.calibrateCamera()."""