# Importing necessary libraries
import cv2
import numpy as np
from sklearn.decomposition import PCA, FastICA

# Load face database
face_database = []
for i in range(len(training_data)):
 face_database.append(cv2.imread('train_faces/{}.png'.format(i+1), cv2.IMREAD_GRAYSCALE))
 face_database = np.array(face_database)

# Mean face vector and transformation matrix using EigenFaces
mean_face = np.mean(face_database, axis=0).reshape(-1, 1)
pca = PCA(n_components=2576)
X = face_database.astype('float32') / 255.
X_train = X
X = pca.fit_transform(X)
eigenvectors = pca.components_
mean_face_pca = mean_face.reshape(-1)

# Projecting training set faces on the face subspace and observing reconstructions
reconstructed_faces = X_train.dot(np.hstack((mean_face_pca.reshape(1, -1), np.identity(2576-len(mean_face_pca)))))
error_faces = face_database - reconstructed_faces
error_faces = error_faces.astype('float32') / 255.
error_faces = pca.transform(error_faces)
print("Error faces:")
for i in range(len(training_data)):
 print("\nError face for image {}:".format(i+1))
 print(error_faces[i])
 print("Orthogonality check:")
 print(np.dot(error_faces, eigenvectors) == 0)

# Nearest-Neighbor (NN) classifier based on the feature vectors
X_test = face_database_test.astype('float32') / 255.
X_test = pca.transform(X_test)
distances = np.linalg.paired_distance(X_train, X_test)
classification = np.argmin(distances, axis=1)
print("Classifications:")
for i in range(len(testing_data)):
 print("\nImage {} is classified as class {}".format(i+1, testing_classes[i], classification[i]))