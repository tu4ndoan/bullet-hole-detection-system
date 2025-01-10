import cv2
import numpy as np

def preprocess_image(image):
    # adjust gamma
    # Step 1: Apply Gamma Correction
    gamma = 1
    alpha = 1
    beta = 0

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    image = cv2.LUT(image, table)

    # Step 2: Apply Contrast and Brightness Adjustment
    
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Step 3: Apply Unsharp Mask for sharpening
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    image =  cv2.addWeighted(image, 1, blurred, 0, 0)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the image to adjust for lighting conditions
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply histogram equalization for better contrast
    #gray = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    _, otsu_thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Edge detection with dynamic thresholds
    median_intensity = np.median(gray)
    lower_thresh = max(0, median_intensity - 150)
    upper_thresh = min(255, median_intensity + 150)
    edges = cv2.Canny(gray, threshold1=lower_thresh, threshold2=upper_thresh)
    
    
    # Apply dilation and erosion to link fragmented edges
    kernel = np.ones((3, 3), np.uint8)  # A 3x3 kernel for dilation and erosion
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)  # Dilate to join edges
    linked_edges = cv2.erode(dilated_edges, kernel, iterations=1)  # Erode to reduce noise

    cv2.imshow("image", image)
    cv2.imshow("edges", thresh)
    cv2.waitKey(0)
    return image, linked_edges



def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

    # 2. Unsharp Masking for sharpening
def unsharp_mask(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # 3. Adaptive Thresholding
def adaptive_threshold(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

    # 5. Contrast and Brightness Adjustment
def adjust_contrast_brightness(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def process_image(image, gamma=1.2, alpha=1.5, beta=30):
    # Step 1: Apply Gamma Correction
    image = adjust_gamma(image, gamma)

    # Step 2: Apply Contrast and Brightness Adjustment
    image = adjust_contrast_brightness(image, alpha, beta)

    # Step 3: Apply Unsharp Mask for sharpening
    image = unsharp_mask(image)

    # Step 4: Apply CLAHE for enhancing local details
    image_clahe = apply_clahe(image)

    # Step 5: Perform Edge Detection (Canny) for fine details
    edges = cv2.Canny(image, 50, 150)

    # Step 6: Adaptive Thresholding for handling high contrast and overexposed areas
    thresh_image = adaptive_threshold(image)

    # Return processed image (you can return all or choose one of the results depending on use)
    return image, image_clahe, edges, thresh_image


def linked_edges(gray_equalized):
    
    # Apply Gaussian Blur to smooth lighting variations
    gray_blurred = cv2.GaussianBlur(gray_equalized, (5, 5), 0)
    
    # Adaptive thresholding to account for varying lighting conditions
    thresh_adaptive = cv2.adaptiveThreshold(
        gray_blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        5
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    clahe_image = clahe.apply(thresh_adaptive)
    # Further binary thresholding if needed
    #ret, inv_thresh = cv2.threshold(gray_blurred, 150, 255, cv2.THRESH_BINARY_INV)
    #_, thresh = cv2.threshold(gray_blurred, 150, 255, cv2.THRESH_BINARY)
    #ret, trunc_thresh = cv2.threshold(clahe_image, 55, 255, cv2.THRESH_TRUNC)
    _, otsu_thresh = cv2.threshold(clahe_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Edge detection with dynamic thresholds
    median_intensity = np.median(gray_blurred)
    lower_thresh = max(0, median_intensity - 50)
    upper_thresh = min(255, median_intensity + 150)
    edges = cv2.Canny(otsu_thresh, threshold1=lower_thresh, threshold2=upper_thresh)
    
    
    # Apply dilation and erosion to link fragmented edges
    kernel = np.ones((3, 3), np.uint8)  # A 3x3 kernel for dilation and erosion
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)  # Dilate to join edges
    linked_edges = cv2.erode(dilated_edges, kernel, iterations=1)  # Erode to reduce noise
    return linked_edges

if __name__ == "__main__":
    image = cv2.imread("./HinhAnh/DaiBan1/BiaSo4-1-9.jpg")
    
    preprocess_image(image)