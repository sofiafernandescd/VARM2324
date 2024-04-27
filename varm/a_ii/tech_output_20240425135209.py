Here is a Python script that solves the exercise task using OpenCV's Deep Neural Network module (dnn) for face detection:

```python
import cv2

# Load the pre-trained model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
net = cv2.dnn.readNetFromTorch(cv2.data.models + 'deploy.prototxt')
net.readweights(cv2.data.models + 'res10-300x300-v1.caffemodel')

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera feed
    ret, img = cap.read()

    # Convert the image to a blob and pass it through the network
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 127))
    net.setInput(blob)
    detections = net.forward()

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting image
    cv2.imshow('Face Detection', img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
```