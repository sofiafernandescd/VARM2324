from sklearn.metrics import pairwise_distances
import numpy as np
import cv2
import matplotlib.pyplot as plt

class EigenfacesModel():
    def __init__(self, m: int = 20) -> None:
        self.m = m
        self.faces = None
        self.mean_face = None
        self.A = None
        self.R = None
        self.V = None
        self.W = None # W cols form and orthonormal basis
        self.projections = None # y = np.dot(W.T, A)
    
    def fit(self, Xtrain, ytrain):

        # transform images into vectors
        faces = []
        for image in Xtrain:
            img = image.copy()
            img = cv2.resize(img, (56, 46))
            faces.append(img.flatten())
        self.faces = np.array(faces)
        print("Faces shape:", self.faces.shape)
        #Faces shape: (2562, 2576)

        # mean face
        self.mean_face = np.mean(self.faces, axis=0)
        self.mean_face = self.mean_face.reshape(((self.mean_face.shape[0], 1)))
        print("Mean Face shape:", self.mean_face.shape) 
        #Mean Face shape: (2576, 1)

        # Compute matrix A containing the "AC components" of the faces from the training set
        self.A = self.faces.T - self.mean_face
        #self.A = self.A.T
        print("A element shape:", self.A[0].shape)
        #A element shape: (2576,)
        
        #self.A = self.A.T
        print("A shape:", self.A.shape)
        #A shape: (2562, 2576)        

        # Compute the eigenvectors and eigenvalues of matrix R = ATA (size of N×N)
        self.R = np.dot(self.A.T, self.A)
        print("R shape:", self.R.shape)
        #R shape: (2576, 2576)
        eigvals, eigvecs = np.linalg.eig(self.R)
        print("eigvals shape:", eigvals.shape, "eigvecs shape:", eigvecs.shape)

        # Select m (maximum of N-1) eigenvectors from R, associated to the highest eigenvalues 
        # and define matrix V (N×m), formed by the m eigenvectors of R
        if self.m==0: # good practice but more expensive
            m = min(eigvecs.shape[0], len(Xtrain) - 1) 
        else:
            m = self.m
        print(m)

        # get top-m eigenvalues indices
        top_m_indices = np.argsort(eigvals)[::-1][:m]
        print("Top m indices:", top_m_indices)
        # get top-m eigevectors (V)
        self.V = eigvecs[:, top_m_indices]
        print("V shape:", self.V.shape)
        #V shape: (2576, 20)

        # Compute the weights for each face
        self.W = np.dot(self.A, self.V)
        print("W shape:", self.W.shape)
        # W shape: (2562, 20) devia ser ()
        self.W_ = self.W/np.linalg.norm(self.W, axis=0)
        print("W_ shape:", self.W_.shape)

        # Compute the projections of each face onto the face subspace
        #self.projections = np.dot(self.W, self.V.T)
        self.projections = np.dot(self.W.T, self.A)
        #self.projections = np.dot(self.W.T, self.W)
        print("Projections shape:", self.projections.shape)

    def predict(self, test_img):

        # resize image
        test_img = cv2.resize(test_img, (46, 56))

        # center image
        centered_test_img = test_img.flatten() - self.mean_face.flatten() 
        print("centered_image shape:", centered_test_img.shape)
        print("V shape: ", self.V.shape)
        print("W shape: ", self.W.shape)

        # Project the test image onto the face subspace
        #test_projection = np.dot(centered_test_img.reshape(1, -1), self.V)
        #test_projection = np.dot(centered_test_img, self.W)
        test_projection = np.dot(self.W.T, centered_test_img)
        print("Test projection shape:", test_projection.shape)

        # Find the index of the closest matching face in the training set
        min_dist = np.inf
        min_index = -1
        # compute euclidean distances of test projection to each train projection
        for i in range(len(self.projections)):
            dist = np.linalg.norm(test_projection - self.projections[:, i])
            # update if a more similar projection is found
            if dist < min_dist:
                min_dist = dist
                min_index = i
              

        # Return the filename of the closest matching face
        return self.faces[min_index].reshape((46, 56)), min_index
    
    def get_error_faces_and_reconstructions(self, indices=[0, 5, 10]):
        
        # Selected already centered samples from A
        selected_faces = self.A[:, indices]
        print("Selected faces:", selected_faces.shape)
        #Selected faces: (2576, 3)
        print("V shape:", self.V.shape)
        #V shape: (2562, 20)

        # Project each centered face onto the face subspace
        selected_projections = np.dot(selected_faces, self.V[indices])
        print("Selected faces projections shape:", selected_projections.shape)
        # Selected faces projections shape: (2576, 20)
        #Resconstructions shape: (2576, 3)
        #Error faces shape: (2576, 3)

        # Reconstruct each face
        #reconstructions = np.dot(self.W, np.dot(self.W.T, self.V.T))
        reconstructions = np.dot(selected_projections, self.V[indices].T) #+ self.mean_face
        #reconstructions = np.dot(self.V.T, selected_faces) + self.mean_face
        #reconstructions = np.dot(self.W.T, selected_faces) + self.mean_face
        print("Resconstructions shape:", reconstructions.shape)
        # Compute the error face for each reconstructed face
        error_faces = selected_faces - reconstructions - self.mean_face
        print("Error faces shape:", error_faces.shape)

        # Verify that the error face is orthogonal to the face subspace
        orthogonality = np.dot(error_faces, self.V[indices])
        print(orthogonality)
        # Verify that the error face is orthogonal to the face subspace
        #orthogonality = np.dot(error_faces.reshape(2576, -1).T, eigenfaces.V)
        print("Orthogonality check:", np.allclose(orthogonality, np.zeros((orthogonality.shape))))

        return error_faces, reconstructions

    def evaluate(self, ytrue, ypred):
        pass

