"""When comparing a shooting target **before** and **after** a shot, it's essential to account for the fact that the target might have shifted slightly due to the impact of the bullet. To handle this, you need to:
1. **Align the images** (accounting for any shifts or rotation) before comparing them.
2. **Subtract or compare the images** to highlight the changes (e.g., the bullet hole).
  
### Steps for Image Alignment and Comparison:

Here's a general outline of how to approach this problem:

1. **Image Preprocessing**: First, read both images (before and after the shot). If the images are already in grayscale, you can skip the conversion to grayscale. If not, convert them to grayscale to focus on intensity differences.
2. **Feature Matching**: Use feature matching techniques (like **ORB**, **SIFT**, or **SURF**) to find keypoints and descriptors between the before and after images. This will allow you to detect and align the images based on common features.
3. **Image Registration**: Once you find matching features, use them to compute a transformation matrix (e.g., homography or affine transformation) to align the images.
4. **Difference Calculation**: Once aligned, subtract the "before" image from the "after" image to highlight the changes.
5. **Post-processing**: Apply thresholding or edge detection to highlight the bullet hole or areas where the target moved.

### Detailed Example:

```python"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import image_processing
import camera
# Load images (before and after the shot) in grayscale

gray_prev = cv2.imread('./Images/Lane1/BiaTest-1-13.jpg',0)
gray_curr = cv2.imread('./Images/Lane1/BiaTest-1-15.jpg',0)
cam_1 = camera.Camera(1,"BiaTest",1)
img = cam_1.capture_image(22)

image_processing.get_elipse_target_center(img)

# Step 1: Feature detection and matching (ORB in this case)
orb = cv2.ORB_create()

# Detect keypoints and descriptors
kp1, des1 = orb.detectAndCompute(gray_prev, None)
kp2, des2 = orb.detectAndCompute(gray_curr, None)

# Use BFMatcher to find the best matches between the descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Sort the good matches based on distance
good_matches = sorted(good_matches, key=lambda x: x.distance)

# Step 2: Draw matches (for visualization)
image_matches = cv2.drawMatches(gray_prev, kp1, gray_curr, kp2, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the matched keypoints
plt.imshow(image_matches)
plt.title('Feature Matches')
plt.show()

# Step 3: Extract matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Step 4: Calculate homography matrix to align the images using RANSAC
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

# Step 5: Warp the 'before' image to align with the 'after' image
height, width = gray_curr.shape
aligned_before = cv2.warpPerspective(gray_prev, M, (width, height))

# Step 6: Calculate the absolute difference between the images
diff_image = cv2.absdiff(aligned_before, gray_curr)

#adaptive = cv2.adaptiveThreshold(diff_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# Step 7: Threshold the difference image to highlight changes (bullet hole)
_, thresh_diff = cv2.threshold(diff_image, 50, 255, cv2.THRESH_BINARY)

# Step 8: Find contours in the thresholded difference image
contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = []
# Step 9: Draw bounding boxes and annotate with numbers
for i, contour in enumerate(contours):
   x, y, w, h = cv2.boundingRect(contour)
   aspect_ratio = float(w) / h if h != 0 else 0
    
   # Calculate the perimeter and area for circularity check
   perimeter = cv2.arcLength(contour, True)
   area = cv2.contourArea(contour)
    
   if perimeter == 0:
      continue
    
   # Calculate circularity
   circularity = 4 * np.pi * area / (perimeter ** 2)
    
   # Condition 1: Aspect ratio should not be too close to 1 (avoid squares and circles)
   # Condition 2: Circularity should not be too close to 1 (avoid circles)
   if 0.2 < aspect_ratio < 2:  # aspect ratio threshold for elongated shapes
   #if True:
      if 0.2 < circularity < 2:
      #if True:
         if 20 < area < 3000:
            filtered_contours.append(contour)
            x, y, w, h = cv2.boundingRect(contour)
            valid_holes.append((x, y, w, h))

            result = {"name": f"{lane}-{turn}",
                    "lane": lane,
                    "turn": turn,
                    "holes": valid_holes
                    }
        
            results.append(result)
   for result in results:
      print(f"loat {result['turn']} ban trung : {len(result['holes'])} phat dan")
    
      for (x, y, w, h) in result['holes']:
         draw_debug(image_curr_turn, x, y, w, result["turn"])
         # Calculate aspect ratio (width/height)
         cv2.rectangle(gray_curr, (x, y), (x + w, y + h), (0, 255, 0), 2)
         cv2.putText(gray_curr, str(f"{int(cv2.contourArea(contour))} - {aspect_ratio}"), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# Step 10: Show the images with bounding boxes and numbers
cv2.imshow('Bullet Holes Detected', gray_curr)
cv2.imshow('Aligned Before Image', aligned_before)
cv2.imshow('Difference Image', diff_image)
cv2.imshow('Thresholded Difference (Bullet Hole)', thresh_diff)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""```

### Explanation of Each Step:
1. **Feature Detection and Matching**: 
   - We use **ORB** (Oriented FAST and Rotated BRIEF), a fast and efficient feature detector and descriptor, to find keypoints and descriptors in both the "before" and "after" images.
   - **BFMatcher** is used to match the descriptors between the two images.
   
2. **Homography Calculation**:
   - `cv2.findHomography()` computes a transformation matrix (homography) based on the matched keypoints to align the two images.
   
3. **Image Warping**:
   - `cv2.warpPerspective()` applies the computed homography matrix to align the "before" image with the "after" image, compensating for any shifts or rotations.
   
4. **Difference Calculation**:
   - The **absolute difference** (`cv2.absdiff()`) is calculated between the aligned "before" image and the "after" image. This highlights the areas where the images differ (e.g., the bullet hole).
   
5. **Thresholding**:
   - **Thresholding** is applied to isolate the areas where significant changes occurred (e.g., the bullet hole).
   
6. **Visualization**:
   - The aligned images, the difference image, and the thresholded difference image are displayed for inspection.

### Fine-Tuning:
- **Feature Detection**: Depending on the quality of the images, you may want to experiment with other feature detection algorithms like **SIFT** (Scale-Invariant Feature Transform) or **SURF** (Speeded-Up Robust Features), which may give better results but are slower and often require licensing for commercial use.
- **Thresholding**: Adjust the threshold value to better highlight the changes. Depending on the noise level, you may also want to apply **Gaussian blur** or **Canny edge detection** to improve the results.

### Notes:
- **Image Alignment**: It's important to ensure that both the "before" and "after" images are captured with minimal movement other than the target's shift due to the bullet.
- **Target Movement**: If the target has moved slightly but not rotated significantly, an affine transformation might be sufficient. For larger movements or rotations, using homography is better.

This process will allow you to align the images and then detect the differences, like the bullet hole or target movement, effectively."""
# this works very well, TODO: dig into it 