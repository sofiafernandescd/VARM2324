import cv2 as cv
import numpy as np

# Load the training set faces
training_set = []
for i in range(1, 31):
    img = cv.imread(f"face_{i}.pgm", cv.IMREAD_GRAYSCALE)
    training_set.append(img.reshape((-1,)))

# Compute the mean face vector and transformation matrix using EigenFaces
mean_vector, eigenvectors = cv.eig(np.cov(training_set))
transformation_matrix = np.hstack((mean_vector.reshape(1, -1), eigenvectors))

# Project some training set faces onto the face subspace and observe their reconstructions
reconstructed_faces = []
for img in training_set:
    projection = np.dot(img, transformation_matrix)
    reconstructed_faces.append(np.dot(transformation_matrix.T, projection))

# Display the original and reconstructed faces
for i in range(len(training_set)):
    print("Original Face:")
    cv.imshow("", np.uint8(255 * training_set[i].reshape((46, 56))))
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Reconstructed Face:")
    cv.imshow("", np.uint8(np.clip(reconstructed_faces[i], 0, 255).reshape((46, 56))))
    cv.waitKey(0)
    cv.destroyAllWindows()

# Compute the error faces (difference between original and reconstructed faces)
error_faces = [np.abs(np.subtract(training_set[i], np.dot(transformation_matrix, np.dot(np.linalg.inv(np.dot(transformation_matrix.T, transformation_matrix)), np.dot(transformation_matrix.T, training_set[i])))) for i in range(len(training_set))]

# Display the error faces
for i in range(len(error_faces)):
    print("Error Face:")
    cv.imshow("", np.uint8(np.clip(np.abs(error_faces[i]).reshape((46, 56)), 0, 255)))
    cv.waitKey(0)
    cv.destroyAllWindows()

# Compute the mean face vector and transformation matrix using FisherFaces
X = np.vstack(training_set)
means, covariances, eigenvectors = cv.pca(X, flags=cv.CV_PCA_VAR_WITH_ZERO_SV)
mean_vector = means.reshape((1, -1))
transformation_matrix = np.hstack((mean_vector, eigenvectors))

# Project some training set faces onto the face subspace and observe their reconstructions using FisherFaces
reconstructed_faces_fisher = []
for img in training_set:
    projection = np.dot(cv.normalize(img - mean_vector, None, 0, 1), eigenvectors)
    reconstructed_faces_fisher.append(np.dot(mean_vector + np.dot(eigenvectors, projection), eigenvectors.T))

# Display the original and reconstructed faces using FisherFaces
for i in range(len(training_set)):
    print("Original Face:")
    cv.imshow("", np.uint8(255 * training_set[i].reshape((46, 56))))
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("Reconstructed Face:")
    cv.imshow("", np.uint8(np.clip(reconstructed_faces_fisher[i].reshape((46, 56)), 0, 255)))
    cv.waitKey(0)
    cv.destroyAllWindows()

# Compute the error faces (difference between original and reconstructed faces) using FisherFaces
error_faces_fisher = [np.abs(np.subtract(training_set[i], np.dot(mean_vector + np.dot(eigenvectors, np.dot(np.linalg.inv(np.dot(np.dot(eigenvectors.T, covariances), eigenvectors)), np.dot(eigenvectors.T, training_set[i])))) for i in range(len(training_set))]

# Display the error faces using FisherFaces
for i in range(len(error_faces_fisher)):
    print("Error Face:")
    cv.imshow("", np.uint8(np.clip(np.abs(error_faces_fisher[i]).reshape((46, 56)), 0, 255)))
    cv.waitKey(0)
    cv.destroyAllWindows()


#This script loads the training set faces, computes the mean face vector and transformation matrix using both EigenFaces and FisherFaces methods, projects some training set faces onto the face subspace, observes their reconstructions, and displays the error faces (difference between original and reconstructed faces).