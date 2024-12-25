import cv2

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read image
image = cv2.imread('./Images/Lane1/BiaSo8-1-0.jpg')
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
ret, frame = cap.read()
if ret:
    image = frame
# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show the image with detected faces
cv2.imshow('Faces Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
