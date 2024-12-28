import image_processing
import camera
import obj_detect

cam_1 = camera.Camera(1,"test", 1)
img = cam_1.capture_image(1)
image_processing.detect_bullet_hole(img, 1, 1, "test", 5, 200, 150, 150, 50, 150, 16, 2, 11)
# tam thoi dung tham so nay
# nhung can phai filter cac lo~ sai
# va xoa background
# va flatten img (kho)