import cv2
import cv2.data
import numpy as np
import os

#face_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_faces(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image using the Haar Cascade Classifier
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=20, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Draw a rectangle around each detected face and return the image with detected faces
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Crop the detected faces from the image
    faces = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]
    return faces[0]

# # Load the face database
# #face_database = [cv2.imread(f"./faces/data/train/{img_file}") for img_file in os.listdir("./faces/data/train/") if img_file.endswith(".jpg")]
# #face_database = [cv2.imread(f"./faces_stor/{img_file}") for img_file in os.listdir("./faces_stor/") if img_file.endswith(".png")]
# face_database = [crop_faces(cv2.imread("face.png")) for i in range(3)]
# #print(face_database)
# # Define the MPEG-7 format dimensions and eye positions
# mpeg7_format = (56, 46)
# eye_positions = [(24, 16), (24, 31)]

# # Normalize each face image in the database
# for face in face_database:
#     # Apply rotation and scaling to normalize the face
#     normalized_face = cv2.resize(face, mpeg7_format)
    
#     # Crop the face to fit the MPEG-7 format
#     cropped_face = normalized_face[10:60, 5:48]

#     #cropped_face = face
    
#     # Detect and align eyes using Haar Cascade classifier
#     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#     #eye_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_mcs_eyepair_big.xml')
#     faces_with_eyes = []
#     for i in range(cropped_face.shape[0]):
#         face_row = cropped_face[i, :]
#         eyes = eye_cascade.detectMultiScale(face_row)
#         if len(eyes) != 0:
#             print(len(eyes))
#         if len(eyes) == 2:
#             # Align both eyes horizontally
#             left_eye_position = tuple(eyes[0])
#             right_eye_position = tuple(eyes[1])
            
#             # Adjust the position of the face to align both eyes
#             adjusted_face = np.copy(cropped_face)
#             adjusted_face[:, left_eye_position[0]:right_eye_position[0]] = cropped_face[:, right_eye_position[0]:left_eye_position[0]]
            
#             # Add the aligned face to the list of faces with eyes properly aligned
#             faces_with_eyes.append(adjusted_face)
            
#             # Save the aligned face image
#             cv2.imwrite(f"./faces/{i}.jpg", adjusted_face)
    
#     # Append the final, normalized and eye-aligned face to the resultant database
#     #face_database.append(faces_with_eyes[0])


#import os
#import cv2
#import numpy as np
#from skimage.transform import resize

# Load the face database (training set)
# face_images = []
# for filename in os.listdir('./faces_stor'):
#     img = cv2.imread(f"./faces_stor/{filename}", 0)  # Read image in grayscale
#     face_images.append(img)


#face_images = [crop_faces(cv2.imread("face.png")) for i in range(3)]



# # Normalize the faces
# normalized_faces = []
# for i, face in enumerate(face_images):
#     # Detect and draw eyes using Haar Cascade classifier
#     left_eye = cv2.CascadeClassifier('./haarcascade/haarcascade_mcs_lefteye.xml')
#     right_eye = cv2.CascadeClassifier('./haarcascade/haarcascade_mcs_righteye.xml')
    
#     face_copy = np.copy(face)
#     left_eyes = left_eye.detectMultiScale(face_copy, scaleFactor=1.3, minNeighbors=20)
#     right_eyes = right_eye.detectMultiScale(face_copy, scaleFactor=1.3, minNeighbors=20)
#     print("Left eyes:", left_eyes)
#     print("Right eyes:", right_eyes)
    
#     # Align eyes horizontally
#     left_x, left_y, left_w, left_h = np.mean([rect[0] for rect in left_eyes], axis=0).astype(int) #np.mean([rect[0] for rect in left_eyes], axis=0).astype(int)
#     right_x, right_y, right_w, right_h = np.mean(right_eyes, axis=0).astype(int) #np.mean([rect[0] for rect in right_eyes], axis=0).astype(int)
    
#     # Crop and resize the face to fit MPEG-7 recommendation
#     x1, y1, x2, y2 = 0, left_y - 5, 46, left_y + right_h + 10
#     cropped_face = face[y1:y2, x1:x2]
    
#     # Resize the cropped face to 56x46
#     normalized_face = cv2.resize(cropped_face, (46, 56))
#     normalized_faces.append(normalized_face)
#     cv2.imwrite(f"normalized_face{i}.png", normalized_face)


import cv2
import numpy as np

# Load Haar Cascade classifier XML files for eyes detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

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
    gray = image.copy() #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=15)

    norm_faces = []
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            normalized_face = preprocess_face(face)
            #yield normalized_face
            norm_faces.append(normalized_face)
    return norm_faces[0]

def main():
    """Load faces from a directory and display the normalized ones."""
    path = 'path/to/faces'
    faces = [cv2.imread("face.png") for i in range(3)]#load_faces(path)

    for face in faces: #detect_faces(cv2.imread('path/to/image.jpg')):
        norm_face = detect_faces(face)
        print("Normalized Face:")
        print(norm_face.shape)
        #cv2.imshow('Normalized Face', cv2.cvtColor(norm_face, cv2.COLOR_GRAY2BGR))
        cv2.imshow('Normalized Face', norm_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
