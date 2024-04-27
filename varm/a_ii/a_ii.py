import cv2
import numpy as np

# Read an image and resize it to 416x416 pixels
img = cv2.imread('image.png')
resized_img = cv2.resize(img, (416, 416))
height, width, _ = img.shape

# Load pre-trained face detection model
net = cv2.dnn.readNetFromDarknet('face_detector.xml')
blob = cv2.dnn.blobFromImage(resized_img, 1.0 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
scores, boxes = net.forward()

# Process the detection results and display the output
for i in np.arange(scores.shape[2]):
    proba = scores[0, 0, i]
    if proba > 0.5:
        (x, y) = (boxes[0, 0, i].astype("int"), boxes[0, 1, i].astype("int"))
        w = int(boxes[0, 2, i] * width)
        h = int(boxes[0, 3, i] * height)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)

import cv2

# Load the classifier
face_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_alt2.xml')

# Read an image and detect faces
img = cv2.imread('image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)