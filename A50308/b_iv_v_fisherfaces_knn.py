import numpy as np
import cv2
from b_iii_v_eigenfaces_knn import EigenfacesModel

class FisherfacesModel():
    def __init__(self, m: int = 20) -> None:
        self.m = m
        self.faces = None
        self.mean_face = None
        self.A = None
        self.R = None
        self.Wpca = None
        self.Wfisher = None
    
    def fit(self, Xtrain, ytrain):
        # transform images into vectors
        faces = []
        for image in Xtrain:
            img = image.copy()
            img = cv2.resize(img, (56, 46))
            faces.append(img.flatten())
        self.faces = np.array(faces)
        print("Faces shape:", self.faces.shape)

        print("EigenfacesModel")
        pca = EigenfacesModel(self.m)
        pca.fit(Xtrain, ytrain)

        # compute mean face
        self.mean_face = pca.mean_face #np.mean(self.faces, axis=0)
        #self.mean_face = self.mean_face.reshape(((self.mean_face.shape[0], 1)))
        print("Mean Face shape:", self.mean_face.shape)

        # subtract mean face from each image
        self.A = pca.A #self.faces.T - self.mean_face
        print("A shape:", self.A.shape)

        
        labels_names = np.unique(ytrain)
        labels = [i for i in range(len(labels_names))]
        
        num_classes = len(labels)
        num_features = 46*56
        num_images = len(Xtrain)

        # Compute the Sb and Sw matrices
        Sb = np.zeros((num_features, num_features))
        Sw = np.zeros((num_features, num_features))
        for i,label in enumerate(labels_names):
    
            class_faces = self.faces[ytrain==label]
            class_mean = np.mean(class_faces, axis=0)
            #print("class_faces:", class_faces.shape)
            #print("class_mean:", class_mean.shape)
            
            class_diff = class_mean - self.mean_face
            #print("class_diff:", class_diff.shape)
            Sb += np.dot(class_diff.T, class_diff)

            class_centered = class_faces - class_mean
            #print("class_centered:", class_centered.shape)
            Sw += np.dot(class_centered.T, class_centered)
           

        # Compute PCA
        self.Wpca = pca.W
        self.Vpca = pca.V

        # Compute Sb and Sw in the PCA subspace
        Sb = np.dot(np.dot(self.Wpca.T, Sb), self.Wpca)
        Sw = np.dot(np.dot(self.Wpca.T, Sw), self.Wpca)
        #Sb = np.dot(np.dot(self.Vpca.T, Sb), self.Vpca)
        #Sw = np.dot(np.dot(self.Vpca.T, Sw), self.Vpca)
        
        # Determine the c-1 “larger eigenvectors” from the matrix Sw-1 Sb (mxm)
        # Compute the Fisherfaces basis
        inv_Sw_Sb = np.dot(np.linalg.inv(Sw), Sb)
        eigenvalues, eigenvectors = np.linalg.eig(inv_Sw_Sb)
        sorted_indices = np.argsort(eigenvalues.real)[::-1]
        #top_m_indices = sorted_indices[:m]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        self.Wfisher = np.dot(self.Wpca, eigenvectors[:, :num_classes - 1])
        # train projections
        self.projections = np.dot(self.Wfisher.T, self.A)
        #self.projections = np.dot(np.dot(self.Wfisher.T, self.Wpca), self.A)
        print("projections shape:", self.projections.shape)


    def predict(self, test_img):

        # resize image
        test_img = cv2.resize(test_img, (46, 56))

        # center image
        centered_test_img = test_img.flatten() - self.mean_face.flatten() 
        print("centered_image shape:", centered_test_img.shape)
        print("Wfisher shape: ", self.Wfisher.shape)

        # Project the test image onto the face subspace
        #test_projection = np.dot(centered_test_img.reshape(1, -1), self.V)
        #test_projection = np.dot(centered_test_img, self.W)
        test_projection = np.dot(self.Wfisher.T, centered_test_img)
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
        print("V shape:", self.Wfisher.shape)
        #V shape: (2562, 20)

        # Project each centered face onto the face subspace
        #selected_projections =np.dot(np.dot(selected_faces, self.Wfisher[indices]), self.Vpca)
        #selected_projections = np.dot(selected_faces, self.Wfisher[indices])
        selected_projections = np.dot(self.Wfisher[indices].T, selected_faces.T)
        print("Selected faces projections shape:", selected_projections.shape)
        # Selected faces projections shape: (2576, 20)
        #Resconstructions shape: (2576, 3)
        #Error faces shape: (2576, 3)

        # Reconstruct each face
        reconstructions = np.dot(selected_projections.T, self.Wfisher[indices].T) #+ self.mean_face
        #reconstructions = np.dot(self.V.T, selected_faces) + self.mean_face
        #reconstructions = np.dot(self.W.T, selected_faces) + self.mean_face
        print("Resconstructions shape:", reconstructions.shape)
        # Compute the error face for each reconstructed face
        error_faces = selected_faces - reconstructions
        print("Error faces shape:", error_faces.shape)
        return error_faces, reconstructions
    

    def evaluate(self, ytrue, ypred):
        pass

