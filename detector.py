# detector.py
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from PIL import Image

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()
        
    def detect_faces(self, image):
        """
        Detect faces in an image using MTCNN
        """
        return self.detector.detect_faces(image)
        
    def extract_and_preprocess_faces(self, input_folder, output_folder):
        """
        Extract faces from images in input folder and save to output folder
        """
        os.makedirs(output_folder, exist_ok=True)
        
        for person_name in os.listdir(input_folder):
            person_path = os.path.join(input_folder, person_name)
            save_path = os.path.join(output_folder, person_name)
            os.makedirs(save_path, exist_ok=True)
            
            if not os.path.isdir(person_path):
                continue  # Skip non-folder files
                
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue  # Skip unreadable images
                    
                # Convert to RGB (MTCNN expects RGB)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                faces = self.detector.detect_faces(image_rgb)
                
                for i, face in enumerate(faces):
                    x, y, w, h = face["box"]
                    x, y = max(0, x), max(0, y)
                    
                    # Crop face
                    face_crop = image_rgb[y:y+h, x:x+w]
                    
                    if face_crop.size == 0:
                        continue  # Skip empty crops
                        
                    # Resize to 160x160 for FaceNet
                    face_crop = Image.fromarray(face_crop).resize((160, 160))
                    
                    # Save cropped face
                    output_path = f"{save_path}/{img_file[:-4]}_{i}.jpg"
                    face_crop.save(output_path)
                    
                    # Draw rectangle around detected face
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                # Show processed image (optional)
                cv2.imshow("Face Detection", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(50)  # Show for 50ms per image
                
        print("✅ Face extraction complete")
        cv2.destroyAllWindows()