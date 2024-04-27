Here is a Python script that solves the exercise task using OpenCV and Numpy libraries for face detection with Haar Cascade Classifier:

```python
import cv2
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    return img

# Load an image
img = cv2.imread('path_to_your_image.jpg')

# Detect faces and display the output
detected_faces_img = detect_faces(img)
cv2.imshow('Detected Faces', detected_faces_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

This script loads a pre-trained Haar Cascade Classifier model to detect faces in an input image. The `detect_faces` function takes an image as input, converts it to grayscale, and then uses the cascade classifier to detect faces. It then draws rectangles around detected faces and returns the output image. Finally, the script loads an image, detects faces in it, and displays the output.