import camera
import cv2
import image_processing
# Load images (before and after the shot) in grayscale

#image_processing.compare_and_detect(1, 14, "BiaTest") # sao loat 15 ko chay?
cam_1 = camera.Camera(1,"BiaSo4",1)
#img1 = cam_1.capture_image(15)
#cv2.waitKey(5000)
#img2 = cam_1.capture_image(1)
image_processing.compare_and_detect(1, 16, "BiaSo4")
# TODO: handle case 2 lo dan gan nhau hoac de len nhau
# TODO: handle case 2 lo dan trung nhau
# TODO: bao diem nhieu dai