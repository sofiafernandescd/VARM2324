```python
import cv2
import numpy as np

# Load Haar Cascade classifier XML files for eyes detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

def load_faces(path):
    """Load faces from a directory and return them as a NumPy array."""
    faces = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces.append(gray)
    return np.array(faces)

def preprocess_face(image):
    """Preprocess a face image to comply with MPEG-7 recommendation."""
    # Convert color image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the face image by scaling and rotating if necessary
    height, width = gray.shape[:2]
    aspect_ratio = float(width) / height

    if aspect_ratio > 1.0:
        k = 1000.0 / width
        d = (height * k - height) / 2.0
        M = np.float32([[k, 0, -d], [0, 1, 0]])
        image = cv2.warpAffine(image, M, (int(width * 0.56), int(height * 0.46)))

    # Crop the face to the required size and format
    x, y, w, h = 30, 30, 46, 56
    image = image[y:y+h, x:x+w]

    return image

def detect_faces(image):
    """Detect faces in an image using Haar Cascade classifier."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            normalized_face = preprocess_face(face)
            yield normalized_face

def main():
    """Load faces from a directory and display the normalized ones."""
    path = 'path/to/faces'
    faces = load_faces(path)

    for face in detect_faces(cv2.imread('path/to/image.jpg')):
        print("Normalized Face:")
        print(face.shape)
        cv2.imshow('Normalized Face', cv2.cvtColor(face, cv2.COLOR_GRAY2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

Replace `path/to/faces` with the path to your face images directory and `path/to/image.jpg` with the path to an image containing faces. This script will load the faces from the specified directory, detect them using Haar Cascade classifier, preprocess them according to MPEG-7 recommendation, and display the normalized faces.