# main.py
import os
import argparse
from detector import FaceDetector
from embedder import FaceEmbedder  
from recognizer import FaceRecognizer

def main():
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['extract', 'train', 'recognize', 'video'],
                        help='Operation mode: extract, train, recognize, or video')
    parser.add_argument('--input', type=str, default='dataset/',
                        help='Input folder for extraction or image for recognition')
    parser.add_argument('--output', type=str, default='outputphoto/',
                        help='Output folder for extracted faces')
    parser.add_argument('--model', type=str, default='face_classifier.pkl',
                        help='Path to save/load the face classifier model')
    parser.add_argument('--video', type=str, default='0',
                        help='Video path or webcam index (default: 0 for webcam)')
                        
    args = parser.parse_args()
    
    if args.mode == 'extract':
        print("Step 1: Face Detection with MTCNN")
        detector = FaceDetector()
        detector.extract_and_preprocess_faces(args.input, args.output)
        
    elif args.mode == 'train':
        print("Step 2: Extract Embeddings using FaceNet (Keras) Then Training SVM for Face Recognition")
        embedder = FaceEmbedder()
        X, y = embedder.prepare_dataset(args.output)
        embedder.train_classifier(X, y, args.model)
        
    elif args.mode == 'recognize':
        print("Step 3: Recognize Faces in New Images")
        recognizer = FaceRecognizer(args.model)
        recognizer.recognize_faces(args.input)
        
    elif args.mode == 'video':
        print("Step 4: Recognize Faces in Video")
        recognizer = FaceRecognizer(args.model)
        # Convert to int if the input is a numeric string (for webcam index)
        video_source = int(args.video) if args.video.isdigit() else args.video
        recognizer.recognize_faces_video(video_source)
    
if __name__ == "__main__":
    main()