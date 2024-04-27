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