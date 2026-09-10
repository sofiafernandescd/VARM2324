import cv2


def detect_faces_dnn(img):

    imgHeight, imgWidth, _ = img.shape

    net = cv2.dnn.readNet("./dnn_models/opencv_face_detector_uint8.pb", "./dnn_models/opencv_face_detector.pbtxt")

    # Convert the image to a blob and pass it through the network
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 127))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>0.5:           
            x1=int(detections[0,0,i,3]*imgWidth)
            y1=int(detections[0,0,i,4]*imgHeight)
            x2=int(detections[0,0,i,5]*imgWidth)
            y2=int(detections[0,0,i,6]*imgHeight)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), int(round(imgHeight/150)), 8)

    return img

def main():
    # Read an image and resize it to 416x416 pixels
    img = cv2.imread('image.png')
    #resized_img = cv2.resize(img, (416, 416))

    # Detect faces in the image
    detected_img = detect_faces_dnn(img)

    # Display the detected faces
    cv2.imshow('Detected Faces', detected_img)
    cv2.waitKey(0)

if __name__ == "__main__":


    # Capture video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera feed
        ret, img = cap.read()

        # Detect faces in the frame
        detected_img = detect_faces_dnn(img)

        # Display the resulting image
        cv2.imshow('Face Detection', img)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()