import cv2
import mediapipe as mp
#import mediapipe.python.solutions.drawing_utils as mp_drawing
#import mediapipe.python.solutions.face_mesh as mp_face_mesh

# Initialize the MediaPipe Pose estimator with the pre-trained model.
mp_drawing = mp.solutions.drawing_utils  # This is used to draw key poses
mp_face_mesh = mp.solutions.face_mesh  # This is the face mesh module

def detect_faces_mediapipe(frame):
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    #results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = face_mesh.process(frame)

    # Draw the face landmarks
    draw_style = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    frame.flags.writeable = True
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = divmod(lm.x * w, w)
                cv2.circle(frame, (int(cx), int(cy)), 5, [255, 0, 0])#, cv2.FILLED)
            
    
    # Display the result
    cv2.imshow('MediaPipe Face Mesh', frame)
    cv2.waitKey(0)

def main():
    # Open the webcam
    img = cv2.imread('face.png')
    detect_faces_mediapipe(img)

if __name__ == "__main__":
    main()