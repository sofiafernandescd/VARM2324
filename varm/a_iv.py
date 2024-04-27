import cv2
import mediapipe as mp

# Initialize MediaPipe's Face Detection Model
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

def detect_faces(image):
    # Create an instance of the model
    face_detection = mp_face_detection.FaceDetection()
    
    # Process the image with the model
    results = face_detection.process(image)
    print(results)
    
    # Draw the detected faces on the image
    #image.draw_landmarks(results.multi_face_landmarks, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
    #results = dict(results)
    print(results['detections'])
    for result in results['detections']:
        mp_drawing.draw_detection(image, result)
    return image

# Initialize the camera and start capturing images
cap = cv2.VideoCapture(0)
while True:
    # Capture a frame from the camera
    _, frame = cap.read()
    # convert to cv2 format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the captured frame
    detected_frame = detect_faces(frame)
    
    # Display the detected frame
    cv2.imshow('Face Detection', detected_frame)
    
    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()