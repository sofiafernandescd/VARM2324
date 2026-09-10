import cv2
import dlib

# Load the pre-trained model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_models/shape_predictor_68_face_landmarks.dat')

def detect_faces_dlib(frame):

    rects = detector(frame, 1)

    for rect in rects:
        # Get facial landmarks
        shape = predictor(frame, rect)
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

        # Draw facial landmarks
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    return frame
    

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Open camera

    while True:
        _, frame = cap.read()
        
        # Detect faces in the image
        frame = detect_faces_dlib(frame)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()