import cv2
import pytesseract

# Set up pytesseract path (if necessary, depending on your installation)
# For example, on Windows, it may look like:
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

# Load the image
image = cv2.imread('./HinhAnh/DaiBan1/BiaSo4-1-0.jpg')

# Preprocess the image (convert to grayscale)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Optional: Apply thresholding to make the image more suitable for OCR
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Use pytesseract to extract text from the image
text = pytesseract.image_to_string(gray)
print(text)
# Check if "10" is in the extracted text
if "10" in text:
    print("Number 10 found!")
else:
    print("Number 10 not found.")
