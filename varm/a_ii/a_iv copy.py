import cv2
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.face_mesh as mp_face_mesh

# Initialize the MediaPipe Pose estimator with the pre-trained model.
#mp_drawing = mp.solutions.drawing_utils  # This is used to draw key poses
#mp_face_mesh = mp.solutions.face_mesh  # This is the face mesh module

def detect_faces_mediapipe(cap):
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # To improve performance, optionally mark the image not writeable to pass by reference.
        frame.flags.writeable = False

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Draw the face landmarks
        draw_style = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = divmod(lm.x * w, w)
                    cv2.circle(frame, (cx, cy), 5, [255, 0, 0], cv2.FILLED)
                
        cv2.imshow('MediaPipe Face Mesh', frame)
        
        # Stop the pipeline if 'q' is pressed.
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    detect_faces_mediapipe(cap)

if __name__ == "__main__":
    main()