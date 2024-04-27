```python
# Import OpenCV library
import cv2

# Load Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Read input image
img = cv2.imread('input_image.jpg')

# Detect faces in the image and display the output
detect_faces(img)
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

This script uses OpenCV library to perform face detection using Haar Cascade Classifier algorithm. The `haarcascade_frontalface_default.xml` file is a pre-trained classifier for detecting frontal faces. The `detect_faces` function takes an input image, converts it to grayscale, and then uses the `CascadeClassifier.detectMultiScale()` method to detect faces in the image. Rectangles are drawn around detected faces and the output is displayed using OpenCV's `imshow()` function.