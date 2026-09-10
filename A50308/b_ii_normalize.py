import cv2
import dlib
import numpy as np
from tqdm import tqdm

# Load face and eye detectors
face_detector = dlib.get_frontal_face_detector()
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define desired image dimensions
IMG_HEIGHT, IMG_WIDTH = 56, 46
DESIRED_LEFT_EYE_X, DESIRED_RIGHT_EYE_X = 16, 31

def normalize_face(img):
    # Convert image to grayscale
    gray = img #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face
    detections = face_detector(gray, 1)
    if len(detections) != 1:
        #raise ValueError('Image must contain exactly one face')
        return img
    
    face = detections[0]

    # Get face square coordinates
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

    # Crop and resize face square
    # By resizing the image before rotating, the distance between the eyes in the 
    # resized image can be used to calculate the angle of rotation required to align 
    # the eyes horizontally. This angle can then be used to rotate the original, un-resized 
    # image to achieve the desired eye alignment.
    # If the image was rotated first, then the distance between the eyes would change due to the rotation, which would make it difficult to accurately calculate the required angle of rotation to achieve the desired eye alignment.
    pad = 1
    face_square = gray[y1-pad:y2+pad, x1-pad:x2+pad] #cv2.resize(gray[y1:y2, x1:x2], (IMG_WIDTH, IMG_HEIGHT))
    #face_square = cv2.resize(gray[y1-pad:y2+pad, x1-pad:x2+pad], (IMG_WIDTH, IMG_HEIGHT))
    #plt.imshow(face_square, cmap='gray')
    
    # Detect eyes in the face square
    eyes = eye_detector.detectMultiScale(face_square)#, scaleFactor=5, minNeighbors=5)
    #print(eyes)
    if len(eyes) != 2:
        # return face_square
        return img #cv2.resize(face_square, (IMG_WIDTH, IMG_HEIGHT))
        #raise ValueError('Both eyes must be detected in the face image')

    # Get left and right eye squares
    left_eye, right_eye = sorted(eyes, key=lambda x: x[0])
    left_eye_x, left_eye_y, left_eye_w, left_eye_h = left_eye
    right_eye_x, right_eye_y, right_eye_w, right_eye_h = right_eye
    #left_eye_square = face_square[left_eye_y:left_eye_y+left_eye_h, left_eye_x:left_eye_x+left_eye_w]
    #right_eye_square = face_square[right_eye_y:right_eye_y+right_eye_h, right_eye_x:right_eye_x+right_eye_w]



    # Calculate the angle between the eyes
    eye_dx = right_eye_x + right_eye_w/2 - (left_eye_x + left_eye_w/2)
    eye_dy = right_eye_y + right_eye_h/2 - (left_eye_y + left_eye_h/2)
    angle = np.degrees(np.arctan2(eye_dy, eye_dx))

    # Calculate the desired coordinates of the left and right eye centers
    #desired_left_eye_center = (DESIRED_LEFT_EYE_X, IMG_HEIGHT//2)
    #desired_right_eye_center = (DESIRED_RIGHT_EYE_X, IMG_HEIGHT//2)
  
    # Calculate the scale factor to fit the face between the eyes and resize face
    #eyes_dist = np.sqrt(eye_dx**2 + eye_dy**2)
    scale = 1 #(desired_right_eye_center[0] - desired_left_eye_center[0]) / eyes_dist
    M = cv2.getRotationMatrix2D((left_eye_x + left_eye_w/2, left_eye_y + left_eye_h/2), angle, scale)
    face_norm = cv2.warpAffine(face_square, M, (face_square.shape[0], face_square.shape[1]))
    face_norm = cv2.resize(face_norm, (IMG_WIDTH, IMG_HEIGHT))
    #plt.imshow(face_norm, cmap='gray')
    # Return normalized face
    return face_norm


if __name__ == "__main__":

    import os

    from b_i_read_images import read_images

    #Prepare the training set consisting of N faces, x1...xN, properly aligned
    faces_path = os.path.join(os.getcwd(), 'faces', 'Original Images', 'Original Images')
    norm_path = os.path.join(os.getcwd(), 'faces', 'Faces', 'Faces_normalized')

    X, Y, file_names = read_images(faces_path)

    # Normalize each face in the training set
    X_norm = []
    for i, img in tqdm(enumerate(X)):
        img_norm = normalize_face(img)
        X_norm.append(img_norm)
        print(img_norm.shape)
        # Save normalized face
        if img_norm.shape == (IMG_HEIGHT, IMG_WIDTH):
            img_name = file_names[i]
            img_path = os.path.join(norm_path, img_name)
            cv2.imwrite(img_path, img_norm)

