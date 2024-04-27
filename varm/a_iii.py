import cv2
import dlib

# Load the pre-trained model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)  # Open camera

while True:
    _, frame = cap.read()
    
    # Detect faces in the image
    rects = detector(frame, 1)

    for rect in rects:
        # Get facial landmarks
        shape = predictor(frame, rect)
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()