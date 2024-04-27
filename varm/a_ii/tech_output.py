# Import OpenCV library and its deep learning module
import cv2.dnn as dnn
import cv2

# Load pre-trained face detection model
model = cv2.dnn.readNetFromDarknet("face_detector.darknet.xml")

# Set input image size for the model
input_size = (416, 416)

def detect_faces(image):
 global model, input_size
 # Pre-process input image
 blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, input_size, (0, 0), True, crop=False)

 # Set input to the model and perform a forward pass
 model.setInput(blob)
 output_layers = model.forward()

 # Get bounding boxes for detected faces
 faces = []
 for output in output_layers:
 for detection in output:
 if detection[0] > 0.5:
 x1, y1, w, h = int(detection[3] * image.shape[1]), int(detection[4] * image.shape[0]), \
 int(w * image.shape[1]), int(h * image.shape[0])
 faces.append((x1, y1, w, h))
 return faces

# Main function to run the script
def main():
 # Load input image
 image = cv2.imread("input_image.jpg")

 # Detect faces in the input image
 faces = detect_faces(image)

 # Draw bounding boxes around detected faces
 for (x1, y1, w, h) in faces:
 cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

 # Display the output image with detected faces
 cv2.imshow("Detected Faces", image)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

if __name__ == "__main__":
 main()