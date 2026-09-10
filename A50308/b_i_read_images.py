import os
import cv2
from tqdm import tqdm


def read_images(faces_path):
    X, Y, file_names = [], [], []
    for img_name in tqdm(os.listdir(faces_path)):
        face_folder = os.path.join(faces_path, img_name)
        for img_pic in os.listdir(face_folder)[:20]:
            file_names.append(img_pic)
            img_path = os.path.join(faces_path, img_name, img_pic)
            #os.rename(img_path, img_path.replace("jpg", "jpeg")) #was throwing error with jpg

            # read image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            X.append(img)
            Y.append(img_path.split('/')[-1].split('_')[0])
    return X, Y, file_names


if __name__=="__main___":

    # Prepare the training set consisting of N faces, x1...xN, properly aligned
    faces_path = os.path.join(os.getcwd(), 'faces', 'Original Images', 'Original Images')

    X, Y, _ = read_images(faces_path)