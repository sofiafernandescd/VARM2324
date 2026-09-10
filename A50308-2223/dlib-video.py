import dlib
import cv2

# Load the pre-trained face detector model from the Dlib library
detector = dlib.get_frontal_face_detector()

# Initialize the camera object and start capturing images
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()

    # Use the face detector model to detect faces in the image
    faces = detector(frame)

    # Draw bounding boxes around the detected faces
    for face in faces:
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Display the annotated images in real-time
    cv2.imshow('Face Detection', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera object and close all windows
camera.release()
cv2.destroyAllWindows()