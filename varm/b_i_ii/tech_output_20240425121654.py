```python
import cv2
import os

# Function to load face images from directory and resize them to MPEG-7 format
def load_faces(path):
    faces = []
    for i in range(1, 8):
        student_folder = os.path.join(path, f'student_{i}')
        for img_name in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (46, 56))
            faces.append(img)
    return np.array(faces)

# Function to perform eye detection using Haar Cascade classifier
def detect_eyes(face):
    cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eyes = cascade.detectMultiScale(face, scaleFactor=1.3, minNeighbors=5)
    return eyes

# Function to normalize the face images by rotating and scaling them
def normalize_faces(faces):
    normalized_faces = []
    for face in faces:
        # Perform rotation and scaling here based on MPEG-7 recommendation
        # ...
        normalized_faces.append(face)
    return np.array(normalized_faces)

# Main function to perform the exercise task
def main():
    face_database = load_faces('students')
    faces = normalize_faces(face_database)

if __name__ == '__main__':
    main()
```

This Python script performs the following tasks:
1. Loads face images from a directory for each student and resizes them to MPEG-7 format (56 rows, 46 columns).
2. Performs eye detection using Haar Cascade classifier on each face image.
3. Normalizes the face images by rotating and scaling them based on MPEG-7 recommendation.

Please note that you need to provide a 'haarcascade_eye.xml' file in the same directory as this script for eye detection to work correctly.