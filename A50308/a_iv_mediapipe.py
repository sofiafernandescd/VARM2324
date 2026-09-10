import cv2
import mediapipe as mp
import numpy as np
#import mediapipe.python.solutions.drawing_utils as mp_drawing
#import mediapipe.python.solutions.face_mesh as mp_face_mesh

# Initialize the MediaPipe Pose estimator with the pre-trained model.
mp_drawing = mp.solutions.drawing_utils  # This is used to draw key poses
mp_face_mesh = mp.solutions.face_mesh  # This is the face mesh module

def detect_faces_mediapipe1(frame):
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
    #cv2.imshow('MediaPipe Face Mesh', frame)
    #cv2.waitKey(0)
    return frame


mp_face_detection = mp.solutions.face_detection

def detect_faces_mediapipe(frame):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
        
        #fonts = cv2.FONT_HERSHEY_PLAIN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detector.process(rgb_frame)
        frame_height, frame_width, c = frame.shape
        if results.detections:
            for face in results.detections:
                face_react = np.multiply(
                    [
                        face.location_data.relative_bounding_box.xmin,
                        face.location_data.relative_bounding_box.ymin,
                        face.location_data.relative_bounding_box.width,
                        face.location_data.relative_bounding_box.height,
                    ],
                    [frame_width, frame_height, frame_width, frame_height]).astype(int)
                
                cv2.rectangle(frame, face_react, color=(255, 255, 255), thickness=2)
                key_points = np.array([(p.x, p.y) for p in face.location_data.relative_keypoints])
                key_points_coords = np.multiply(key_points,[frame_width,frame_height]).astype(int)
                for p in key_points_coords:
                    cv2.circle(frame, p, 4, (255, 255, 255), 2)
                    cv2.circle(frame, p, 2, (0, 0, 0), -1)
        
        return frame


def main():
    # Open the webcam
    img = cv2.imread('face.png')
    detect_faces_mediapipe(img)

if __name__ == "__main__":
    #main()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        
        # Detect faces in the captured frame
        detected_frame = detect_faces_mediapipe(frame)
        
        # Display the detected frame
        cv2.imshow('Face Detection', detected_frame)
        
        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
