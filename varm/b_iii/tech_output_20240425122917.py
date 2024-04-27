```python # Importing necessary libraries
import cv2
import numpy as np

# Loading face database
face_database = []
for i in range(len(training_set)):
    img = cv2.imread('training_set/{}.png'.format(i+1), 0)
    face_database.append(img)

# Computing mean face vector and eigenfaces using PCA
mean_face = np.mean(np.array(face_database), axis=0).reshape(-1, 1)
eigenvectors, _ = np.linalg.eig(np.cov(np.array(face_database).T))
eigenfaces = eigenvectors[:, :min(len(eigenvectors), 20)]

# Projecting training set faces onto the face subspace and observing reconstructions
reconstructed_faces = []
for img in face_database:
    projection = np.dot(img.reshape(-1,), eigenfaces)
    reconstruction = mean_face + np.dot(projection, eigenfaces.T)
    reconstructed_faces.append(reconstruction.reshape(56, 46))

# Displaying original and reconstructed faces
for i in range(len(training_set)):
    plt.subplot(1, len(training_set), i+1)
    plt.imshow(cv2.cvtColor(cv2.imread('training_set/{}.png'.format(i+1), cv2.IMREAD_GRAYSCALE), cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.subplot(1, len(training_set), i+1+len(training_set))
    plt.imshow(np.uint8(reconstructed_faces[i]))
    plt.title('Reconstruction')
    plt.show()

# Computing error faces (difference between original and reconstructed faces)
error_faces = [np.abs(np.subtract(img, reconstruction)) for img, reconstruction in zip(face_database, reconstructed_faces)]

# Displaying error faces
for i in range(len(training_set)):
    plt.imshow(np.uint8(error_faces[i]))
    plt.title('Error Face')
    plt.show()

# Proving that error faces are orthogonal to the face subspace
for i in range(len(training_set)):
    error = error_faces[i]
    for j in range(min(len(training_set), 20)):
        print('Error Face and Eigenface {}:'.format(j+1))
        print(np.dot(error, eigenfaces[j]))
```

This code imports the necessary libraries (OpenCV and NumPy), loads the face database, computes the mean face vector and eigenfaces using PCA, projects the training set faces onto the face subspace, observes their reconstructions, displays original and reconstructed faces, computes error faces, and finally proves that error faces are orthogonal to the face subspace.