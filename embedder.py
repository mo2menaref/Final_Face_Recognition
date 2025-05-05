# embedder.py
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import os
from PIL import Image

class FaceEmbedder:
    def __init__(self):
        # Initialize FaceNet
        self.embedder = FaceNet()
        self.facenet_model = self.embedder.model
        
    def get_embedding(self, face_pixels):
        """
        Get the face embedding for given face pixels
        """
        # Convert to float32
        face_pixels = face_pixels.astype('float32')
        
        # Standardize 
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        
        # Expand dimensions for model
        face_pixels = np.expand_dims(face_pixels, axis=0)
        
        # Get embedding
        embedding = self.facenet_model.predict(face_pixels)[0]
        return embedding
        
    def prepare_dataset(self, dataset_path):
        """
        Prepare dataset for training by extracting embeddings
        """
        X, y = [], []
        
        for person_name in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person_name)
            
            if not os.path.isdir(person_folder):
                continue  # Skip non-folder files
                
            for img_file in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_file)
                
                # Load and preprocess face image
                image = Image.open(img_path).resize((160, 160))
                image = np.array(image)
                
                # Get embedding
                embedding = self.get_embedding(image)
                
                # Store data
                X.append(embedding)
                y.append(person_name)
                
        # Convert to NumPy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Normalize embeddings
        normalizer = Normalizer(norm="l2")
        X = normalizer.transform(X)
        
        return X, y
        
    def train_classifier(self, X, y, model_save_path="face_classifier.pkl"):
        """
        Train SVM classifier and save model
        """
        # Train SVM classifier
        classifier = SVC(kernel="linear", probability=True)
        classifier.fit(X, y)
        
        # Save classifier
        with open(model_save_path, "wb") as f:
            pickle.dump(classifier, f)
            
        print(f"✅ Face recognition model trained and saved as '{model_save_path}'.")
        
        return classifier