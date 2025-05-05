# recognizer.py
import cv2
import os
import numpy as np
from PIL import Image
import pickle
from detector import FaceDetector
from embedder import FaceEmbedder

class FaceRecognizer:
    def __init__(self, model_path="face_classifier.pkl"):
        """
        Initialize face recognition system
        """
        # Load MTCNN detector
        self.detector = FaceDetector()
        
        # Load FaceNet embedder
        self.embedder = FaceEmbedder()
        
        # Load classifier
        with open(model_path, "rb") as f:
            self.classifier = pickle.load(f)
            
    def recognize_faces(self, image_path):
        """
        Recognize faces in an image
        """
        image = cv2.imread(image_path)
        
        if image is None:
            print("❌ Error: Image not found!")
            return
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = self.detector.detector.detect_faces(image_rgb)
        
        for box in boxes:
            x, y, w, h = box["box"]
            x, y = max(0, x), max(0, y)
            
            face_crop = image_rgb[y:y+h, x:x+w]
            face_crop = Image.fromarray(face_crop).resize((160, 160))
            face_crop = np.array(face_crop)
            
            # Get embedding using keras-facenet
            embedding = self.embedder.get_embedding(face_crop)
            embedding = np.expand_dims(embedding, axis=0)
            
            # Predict person name
            person_name = self.classifier.predict(embedding)[0]
            prob = self.classifier.predict_proba(embedding).max()
            
            # Draw rectangle and name
            color = (0, 255, 0) if prob > 0.8 else (0, 0, 255)
            display_name = person_name if prob > 0.8 else "Unknown"
            
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, f"{display_name} ({prob:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Face Recognition", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    def recognize_faces_video(self, video_path=0):
        """
        Recognize faces in video (file or webcam)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Failed to open video source: {video_path}")
            return
            
        print("🎦 Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = self.detector.detector.detect_faces(image_rgb)
            
            for box in boxes:
                x, y, w, h = box["box"]
                x, y = max(0, x), max(0, y)
                
                face_crop = image_rgb[y:y+h, x:x+w]
                face_crop = Image.fromarray(face_crop).resize((160, 160))
                face_crop = np.array(face_crop)
                
                embedding = self.embedder.get_embedding(face_crop)
                embedding = np.expand_dims(embedding, axis=0)
                
                person_name = self.classifier.predict(embedding)[0]
                prob = self.classifier.predict_proba(embedding).max()
                
                color = (0, 255, 0) if prob > 0.8 else (0, 0, 255)
                display_name = person_name if prob > 0.8 else "Unknown"
            
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{display_name} ({prob:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
            cv2.imshow("Live Face Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        cap.release()
        cv2.destroyAllWindows()