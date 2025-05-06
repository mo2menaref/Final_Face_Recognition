# Face Recognition System - User Guide

## Overview

This face recognition system uses MTCNN for face detection, FaceNet for embedding extraction, and SVM for classification. It can detect, register, and recognize faces in images, videos, or through a webcam.

## System Requirements

- **Python Version**: 3.7 to 3.10 (3.9.0 recommended)
- **Required Libraries**:
  - OpenCV (cv2)
  - MTCNN
  - TensorFlow/Keras
  - scikit-learn
  - PIL (Pillow)
  - Flask (for web application)
  - NumPy

## Project Structure

```
face-recognition/
├── dataset/                  # Source images for training (organized by person)
│   ├── person1/              # Folder named after the person
│   │   ├── image1.jpg        # Source images of person1
│   │   └── ...
│   └── person2/
│       └── ...
├── models/                   # Additional model files
│   ├── face_recognition_model.* # FaceNet model used for embeddings
│   └── normalizer.pkl        # Normalizer for face embeddings
├── outputphoto/              # Processed face images (created during extraction)
│   ├── person1/              
│   └── person2/
├── templates/                # Flask templates (created when running simple_app.py)
│   ├── login.html
│   └── dashboard.html
├── _pycache_/                # Python cache files (auto-generated)
├── detector.py               # Face detection module using MTCNN
├── embedder.py               # Face embedding extraction using FaceNet
├── recognizer.py             # Face recognition implementation
├── main.py                   # Command-line interface
├── simple_app.py             # Web application interface
├── face_classifier.pkl       # SVM classifier model (created after training)
├── README.md                 # Project documentation
└── test*.jpeg                # Test images for recognition
```

## Quick Start Guide

### Step 1: Install Dependencies

```bash
pip install opencv-python mtcnn keras-facenet pillow scikit-learn flask numpy
```

### Step 2: Prepare Your Dataset

Organize your face images in the following structure:
```
dataset/
├── John/
│   ├── john_1.jpg
│   ├── john_2.jpg
│   └── ...
├── Alice/
│   ├── alice_1.jpg
│   └── ...
└── ...
```

- Each person should have their own folder named after them
- Include multiple images of each person from different angles for better accuracy
- Ensure faces are clearly visible in the images

### Step 3: Extract Faces

Extract and preprocess faces from your dataset:

```bash
python main.py --mode extract --input dataset/ --output outputphoto/
```

This will:
- Detect faces in each image using MTCNN
- Crop and resize faces to 160x160 pixels
- Save processed faces to the outputphoto/ directory

### Step 4: Train the Model

Train the face recognition model using the extracted faces:

```bash
python main.py --mode train --output outputphoto/ --model face_classifier.pkl
```

This will:
- Extract embeddings from processed faces using FaceNet
- Train an SVM classifier on these embeddings
- Save the trained model as face_classifier.pkl

### Step 5: Recognition

#### Recognize faces in an image:
```bash
python main.py --mode recognize --input path/to/your/image.jpg --model face_classifier.pkl
```

#### Recognize faces in a video:
```bash
python main.py --mode video --video path/to/your/video.mp4 --model face_classifier.pkl
```

#### Recognize faces using webcam:
```bash
python main.py --mode video --video 0 --model face_classifier.pkl
```

### Step 6: Web Application

The project includes a web-based interface with registration and login capabilities:

```bash
python simple_app.py
```

Open your browser and navigate to http://127.0.0.1:5000/ to access the application.

## Customizing the System

### Changing Dataset Location

To use a different dataset location, modify the `--input` parameter in main.py:

```bash
python main.py --mode extract --input your_custom_dataset/ --output outputphoto/
```

### Changing Output Location

To save processed faces to a different location, modify the `--output` parameter:

```bash
python main.py --mode extract --input dataset/ --output your_custom_output/
```

Don't forget to use the same output folder when training:

```bash
python main.py --mode train --output your_custom_output/ --model face_classifier.pkl
```

### Changing Model Name/Location

To save the model with a different name or location:

```bash
python main.py --mode train --output outputphoto/ --model your_model_name.pkl
```

And when using it:

```bash
python main.py --mode recognize --input test.jpg --model your_model_name.pkl
```

## Advanced Configuration

### Models Folder

The `models/` directory contains important model files used by the face recognition system:

- **face_recognition_model**: This is the FaceNet model used for generating face embeddings. FaceNet converts face images into 128-dimensional vectors that represent the unique features of each face.

- **normalizer.pkl**: This is a scikit-learn Normalizer object that standardizes face embeddings to ensure consistent scaling. Proper normalization improves classification accuracy.

These models are used by the `embedder.py` module when converting face images to embeddings. The system loads these models automatically, so you typically won't need to modify them unless you want to use alternative models.

### Face Detection Parameters

The MTCNN detector in `detector.py` can be tuned for different scenarios. Edit the `detector.py` file to adjust:

- Detection thresholds
- Face size parameters
- Additional preprocessing steps

### Recognition Threshold

In `recognizer.py`, you can adjust the confidence threshold for face recognition:

```python
# Current setting:
color = (0, 255, 0) if prob > 0.8 else (0, 0, 255)
display_name = person_name if prob > 0.8 else "Unknown"
```

Lower the threshold (e.g., 0.6) for more lenient recognition or increase it (e.g., 0.9) for stricter matching.

## Troubleshooting

### Model Not Found Error

If you get a "Model not found" error, ensure:
- You have trained the model using the `--mode train` option
- You are specifying the correct path to the model with `--model`

### No Faces Detected

If the system is not detecting faces:
- Check that your images contain clear, well-lit faces
- Try different images from various angles
- Ensure the person is facing the camera

### Poor Recognition Accuracy

If recognition accuracy is poor:
- Add more training images for each person (10+ per person recommended)
- Include images with different lighting, angles, and expressions
- Re-train the model after adding more training data

### Web Application Issues

If you encounter issues with the web application:
- Check that all required libraries are installed
- Ensure your webcam is working properly
- Check browser console for any JavaScript errors

## Performance Tips

- For better accuracy, include multiple training images (10+) per person
- Ensure training images have good lighting and clear views of faces
- Use a variety of facial expressions and angles in training images
- Higher resolution images can improve recognition accuracy
- The system performs best with frontal or slightly angled faces

## Command Reference

```
python main.py --mode [extract|train|recognize|video] [options]

Options:
  --mode      Operation mode: extract, train, recognize, or video
  --input     Input folder for extraction or image for recognition
  --output    Output folder for extracted faces
  --model     Path to save/load the face classifier model
  --video     Video path or webcam index (default: 0 for webcam)
```

## Web Application Features

The web application (`simple_app.py`) provides:

1. **User Registration**: Register new faces in the system
2. **Face Login**: Login using face recognition
3. **User Dashboard**: View recognition results and system status
4. **Real-time Feedback**: Live camera feed with face detection
5. **Multi-User Support**: Register and recognize multiple users