# enhanced_simple_app.py - Face recognition with registration and detailed verification
from flask import Flask, render_template, Response, redirect, url_for, session, flash, jsonify, request
import cv2
import os
import time
import numpy as np
from PIL import Image
import pickle
from datetime import datetime
from pathlib import Path
import shutil

# First, create template files
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)

# Create dataset directory for face registration
dataset_dir = Path(__file__).parent / "dataset"
dataset_dir.mkdir(exist_ok=True)

# Create processed faces directory
faces_dir = Path(__file__).parent / "outputphoto"
faces_dir.mkdir(exist_ok=True)

# Write login.html template - Modified to include registration option
login_html = """<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition System</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 30px; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        .camera { width: 640px; height: 480px; margin: 20px auto; border: 1px solid #ccc; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; 
                border: none; cursor: pointer; margin: 10px; }
        .register-btn { background-color: #2196F3; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .success { background-color: #dff0d8; color: #3c763d; }
        .error { background-color: #f2dede; color: #a94442; }
        .processing { background-color: #fcf8e3; color: #8a6d3b; }
        .tabs { display: flex; justify-content: center; margin-bottom: 20px; }
        .tab { padding: 10px 20px; cursor: pointer; background-color: #f1f1f1; border: 1px solid #ccc; }
        .tab.active { background-color: #e9e9e9; font-weight: bold; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .input-group { margin: 20px 0; }
        input[type="text"] { padding: 10px; width: 300px; }
        .recognized-users { margin-top: 20px; padding: 15px; background-color: #f8f8f8; border-radius: 5px; }
        .user-item { display: inline-block; margin: 5px; padding: 5px 10px; background-color: #e0e0e0; border-radius: 3px; }
        .user-recognized { background-color: #c8e6c9; }
        .user-not-recognized { background-color: #ffcdd2; text-decoration: line-through; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Recognition System</h1>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('login')">Login</div>
            <div class="tab" onclick="switchTab('register')">Register</div>
        </div>
        
        <div id="message-container"></div>
        
        <div class="camera">
            <img id="videoElement" src="{{ url_for('video_feed') }}" alt="Video Feed">
        </div>
        
        <div id="login-tab" class="tab-content active">
            <p>Look at the camera and click the button to verify your face.</p>
            <button id="loginButton">Verify Face</button>
            
            <div id="recognized-users" class="recognized-users" style="display: none;">
                <h3>Verification Results</h3>
                <div id="users-list"></div>
            </div>
        </div>
        
        <div id="register-tab" class="tab-content">
            <div class="input-group">
                <input type="text" id="username" placeholder="Enter your name">
            </div>
            <p>Look at the camera and click the button to register your face.</p>
            <button id="registerButton" class="register-btn">Register Face</button>
            <div id="capture-progress" style="display: none;">
                <p>Capturing... <span id="capture-count">0</span>/10</p>
                <progress id="capture-progress-bar" value="0" max="10" style="width: 300px;"></progress>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabName) {
            // Hide all tabs and remove active class
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab and set active class
            document.getElementById(tabName + '-tab').classList.add('active');
            document.querySelector(`.tab[onclick="switchTab('${tabName}')"]`).classList.add('active');
            
            // Clear messages
            document.getElementById('message-container').innerHTML = '';
            document.getElementById('recognized-users').style.display = 'none';
        }
        
        document.getElementById('loginButton').addEventListener('click', function() {
            // Show processing message
            document.getElementById('message-container').innerHTML = 
                '<div class="message processing">Taking snapshot and verifying...</div>';
            
            // Send request to capture and verify
            fetch('/capture_and_verify')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        document.getElementById('message-container').innerHTML = 
                            '<div class="message success">Welcome, ' + data.user + '!</div>';
                        
                        // Display recognized and not recognized users
                        displayRecognizedUsers(data.all_users, data.recognized_users);
                        
                        // Redirect after showing results
                        setTimeout(() => { window.location.href = '/dashboard'; }, 3000);
                    } else {
                        document.getElementById('message-container').innerHTML = 
                            '<div class="message error">Face not recognized</div>';
                            
                        // Display recognized and not recognized users
                        if (data.all_users && data.recognized_users) {
                            displayRecognizedUsers(data.all_users, data.recognized_users);
                        }
                    }
                });
        });
        
        function displayRecognizedUsers(allUsers, recognizedUsers) {
            const usersList = document.getElementById('users-list');
            usersList.innerHTML = '';
            
            allUsers.forEach(user => {
                const userItem = document.createElement('div');
                userItem.textContent = user;
                userItem.className = 'user-item';
                
                if (recognizedUsers.includes(user)) {
                    userItem.classList.add('user-recognized');
                } else {
                    userItem.classList.add('user-not-recognized');
                }
                
                usersList.appendChild(userItem);
            });
            
            document.getElementById('recognized-users').style.display = 'block';
        }
        
        document.getElementById('registerButton').addEventListener('click', function() {
            const username = document.getElementById('username').value.trim();
            
            if (!username) {
                document.getElementById('message-container').innerHTML = 
                    '<div class="message error">Please enter your name</div>';
                return;
            }
            
            // Show progress indicators
            document.getElementById('capture-progress').style.display = 'block';
            document.getElementById('capture-count').textContent = '0';
            document.getElementById('capture-progress-bar').value = 0;
            
            // Show processing message
            document.getElementById('message-container').innerHTML = 
                '<div class="message processing">Starting face registration...</div>';
            
            // Variables to track progress
            let captureCount = 0;
            const totalCaptures = 10;
            
            // Function to capture faces
            function captureFace() {
                fetch('/capture_face', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ username: username })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        captureCount++;
                        document.getElementById('capture-count').textContent = captureCount;
                        document.getElementById('capture-progress-bar').value = captureCount;
                        
                        if (captureCount < totalCaptures) {
                            // Continue capturing
                            setTimeout(captureFace, 1000); // 1 second delay between captures
                        } else {
                            // All captures complete, train model
                            document.getElementById('message-container').innerHTML = 
                                '<div class="message processing">Training face recognition model...</div>';
                            
                            fetch('/train_model')
                                .then(response => response.json())
                                .then(data => {
                                    if (data.status === 'success') {
                                        document.getElementById('message-container').innerHTML = 
                                            '<div class="message success">Registration successful! You can now login.</div>';
                                    } else {
                                        document.getElementById('message-container').innerHTML = 
                                            '<div class="message error">Error training model: ' + data.message + '</div>';
                                    }
                                    document.getElementById('capture-progress').style.display = 'none';
                                });
                        }
                    } else {
                        document.getElementById('message-container').innerHTML = 
                            '<div class="message error">Error capturing face: ' + data.message + '</div>';
                        document.getElementById('capture-progress').style.display = 'none';
                    }
                });
            }
            
            // Start capturing faces
            captureFace();
        });
    </script>
</body>
</html>
"""

# Write dashboard.html template
dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <title>Dashboard</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        .welcome { background-color: #e8f5e9; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .button { background-color: #f44336; color: white; padding: 10px 20px; 
                border: none; cursor: pointer; text-decoration: none; display: inline-block; margin: 10px; }
        .home-btn { background-color: #2196F3; }
        .check { color: green; font-weight: bold; }
        .user-list { margin: 20px 0; padding: 15px; background-color: #f8f8f8; border-radius: 5px; text-align: left; }
        .user-item { display: inline-block; margin: 5px; padding: 5px 10px; background-color: #e0e0e0; border-radius: 3px; }
        .user-recognized { background-color: #c8e6c9; }
        .user-not-recognized { background-color: #ffcdd2; text-decoration: line-through; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dashboard</h1>
        <div class="welcome">
            <h2>Welcome, {{ username }}!</h2>
            <p>You have successfully logged in using face recognition.</p>
        </div>
        <div class="user-list">
            <h3>Users in System</h3>
            {% for user in all_users %}
                <div class="user-item {% if user in recognized_users %}user-recognized{% else %}user-not-recognized{% endif %}">
                    {{ user }}
                </div>
            {% endfor %}
        </div>
        <div>
            <h3>System Status</h3>
            <p><span class="check">&#10004;</span> Face recognition system is active</p>
            <p><span class="check">&#10004;</span> Last login: {{ now }}</p>
            <p><span class="check">&#10004;</span> Total registered users: {{ all_users|length }}</p>
        </div>
        <a href="/" class="button home-btn">Home</a>
        <a href="/logout" class="button">Logout</a>
    </div>
</body>
</html>
"""

# Write templates to files with utf-8 encoding
with open(templates_dir / "login.html", "w", encoding="utf-8") as f:
    f.write(login_html)

with open(templates_dir / "dashboard.html", "w", encoding="utf-8") as f:
    f.write(dashboard_html)

print(f"Templates created in: {templates_dir}")

# Create Flask app
app = Flask(__name__, template_folder=str(templates_dir))
app.secret_key = "face_recognition_login_secret"

# Load face recognition components
try:
    from detector import FaceDetector
    from embedder import FaceEmbedder
    
    detector = FaceDetector()
    embedder = FaceEmbedder()
    
    # Load classifier if it exists
    classifier_path = "face_classifier.pkl"
    if os.path.exists(classifier_path):
        with open(classifier_path, "rb") as f:
            classifier = pickle.load(f)
        print("Face recognition model loaded successfully")
    else:
        classifier = None
        print("No face recognition model found")
        
    # Function to get all registered users
    def get_all_users():
        if os.path.exists(faces_dir):
            return [dir_name for dir_name in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, dir_name))]
        return []
    
    # Initial list of authorized users
    ALL_USERS = get_all_users()
    if not ALL_USERS:
        ALL_USERS = ["Nabil", "Moamen", "Mohamed"]  # Default users if none registered
    
except Exception as e:
    print(f"Error loading face recognition components: {e}")
    print("Using mock implementation for testing purposes")
    # Mock implementation for testing
    class MockDetector:
        def detect_faces(self, image):
            h, w = image.shape[:2]
            return [{"box": [w//4, h//4, w//2, h//2]}]
    
    detector = MockDetector()
    classifier = None
    ALL_USERS = ["Nabil", "Moamen", "Mohamed"]

# Global variables
camera = None
current_frame = None  # Store the current frame for snapshot

def generate_frames():
    global camera, current_frame
    
    if camera is None:
        try:
            camera = cv2.VideoCapture(0)
        except Exception as e:
            print(f"Error opening camera: {e}")
            # Return empty frames if camera fails
            while True:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    while True:
        try:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame from camera")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Store current frame for later capture
                current_frame = frame.copy()
                
                # Process for display only (show detected faces without verification)
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Display mode with identification when possible
                    boxes = detector.detect_faces(image_rgb) if hasattr(detector, 'detect_faces') else detector.detector.detect_faces(image_rgb)
                    
                    for box in boxes:
                        x, y, w, h = box["box"]
                        x, y = max(0, x), max(0, y)
                        
                        # If classifier exists, attempt to identify the face
                        if classifier is not None:
                            # Extract and process face
                            if y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                                face_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[y:y+h, x:x+w]
                                if face_crop.size > 0:
                                    face_crop = Image.fromarray(face_crop).resize((160, 160))
                                    face_crop = np.array(face_crop)
                                    
                                    # Get embedding and predict
                                    embedding = embedder.get_embedding(face_crop)
                                    embedding = np.expand_dims(embedding, axis=0)
                                    
                                    person_name = classifier.predict(embedding)[0]
                                    prob = classifier.predict_proba(embedding).max()
                                    
                                    # Display based on confidence threshold
                                    color = (0, 255, 0) if prob > 0.6 else (0, 0, 255)
                                    display_name = person_name if prob > 0.6 else "Unknown"
                                    
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                                    cv2.putText(frame, f"{display_name} ({prob:.2f})", 
                                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                    continue
                        
                        # Default display if no classifier or face processing failed
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                        cv2.putText(frame, "Face Detected", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                                
                except Exception as e:
                    print(f"Error in face detection: {e}")
                    cv2.putText(frame, "Detection Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def verify_face(frame):
    """Verify if the face in the frame belongs to an authorized user"""
    global ALL_USERS
    
    try:
        if frame is None:
            print("No frame available for verification")
            return None, []
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # For testing mode
        if classifier is None:
            # Mock recognition (for testing)
            boxes = detector.detect_faces(image_rgb) if hasattr(detector, 'detect_faces') else []
            if boxes:
                # Randomly pick a user for testing
                import random
                recognized = [random.choice(ALL_USERS)]
                return recognized[0], recognized
            return None, []
            
                    # Actual recognition
        boxes = detector.detector.detect_faces(image_rgb)
        recognized_users = []
        primary_user = None
        max_prob = 0
        
        for box in boxes:
            x, y, w, h = box["box"]
            x, y = max(0, x), max(0, y)
            
            if y+h > frame.shape[0] or x+w > frame.shape[1]:
                continue
                
            face_crop = image_rgb[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
                
            face_crop = Image.fromarray(face_crop).resize((160, 160))
            face_crop = np.array(face_crop)
            
            embedding = embedder.get_embedding(face_crop)
            embedding = np.expand_dims(embedding, axis=0)
            
            person_name = classifier.predict(embedding)[0]
            prob = classifier.predict_proba(embedding).max()
            
            # Display face information on the frame
            color = (0, 255, 0) if prob > 0.6 else (0, 0, 255)
            display_name = person_name if prob > 0.6 else "Unknown"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{display_name} ({prob:.2f})", (x, y-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if prob > 0.6 and person_name in ALL_USERS:
                recognized_users.append(person_name)
                if prob > max_prob:
                    primary_user = person_name
                    max_prob = prob
                
        return primary_user, recognized_users
                
    except Exception as e:
        print(f"Error in face verification: {e}")
        return None, []

def capture_face_for_registration(frame, username):
    """Capture a face for registration"""
    try:
        if frame is None:
            return False, "No frame available"
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes = detector.detector.detect_faces(image_rgb) if hasattr(detector, 'detector') else detector.detect_faces(image_rgb)
        
        if not boxes:
            return False, "No face detected"
            
        # Use the largest face (assume it's the user)
        largest_box = max(boxes, key=lambda box: box["box"][2] * box["box"][3])
        x, y, w, h = largest_box["box"]
        x, y = max(0, x), max(0, y)
        
        if y+h > frame.shape[0] or x+w > frame.shape[1]:
            return False, "Face out of frame"
            
        face_crop = image_rgb[y:y+h, x:x+w]
        if face_crop.size == 0:
            return False, "Invalid face crop"
            
        # Create user directory if it doesn't exist
        user_dir = dataset_dir / username
        user_dir.mkdir(exist_ok=True)
        
        # Also create processed faces directory for this user
        processed_dir = faces_dir / username
        processed_dir.mkdir(exist_ok=True)
        
        # Save the face image
        timestamp = int(time.time() * 1000)
        img_path = user_dir / f"{username}_{timestamp}.jpg"
        
        # Resize and save the face
        face_img = Image.fromarray(face_crop)
        face_img.resize((160, 160)).save(str(img_path))
        
        # Also save the processed face
        processed_img_path = processed_dir / f"{username}_{timestamp}.jpg"
        face_img.resize((160, 160)).save(str(processed_img_path))
        
        return True, str(img_path)
        
    except Exception as e:
        print(f"Error capturing face: {e}")
        return False, str(e)

def train_model():
    """Train the face recognition model with the processed faces"""
    global classifier, ALL_USERS
    
    try:
        # Create FaceEmbedder
        embedder = FaceEmbedder()
        
        # Prepare dataset from processed faces
        X, y = embedder.prepare_dataset(str(faces_dir))
        
        if len(X) == 0 or len(y) == 0:
            return False, "No face data available for training"
            
        # Train classifier
        embedder.train_classifier(X, y, "face_classifier.pkl")
        
        # Load the newly trained classifier
        with open("face_classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
            
        # Update the list of authorized users
        ALL_USERS = get_all_users()
        
        return True, "Model trained successfully"
        
    except Exception as e:
        print(f"Error training model: {e}")
        return False, str(e)

@app.route('/')
def index():
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_and_verify')
def capture_and_verify():
    global current_frame, ALL_USERS
    
    if current_frame is None:
        return jsonify({'status': 'fail', 'message': 'No camera feed available'})
    
    # Take snapshot and verify
    primary_user, recognized_users = verify_face(current_frame)
    
    # Update the list of all users
    if os.path.exists(faces_dir):
        ALL_USERS = [dir_name for dir_name in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, dir_name))]
    
    if len(ALL_USERS) == 0:
        ALL_USERS = ["Nabil", "Moamen", "Mohamed"]  # Default users if none registered
    
    if primary_user in ALL_USERS:
        session['logged_in'] = True
        session['username'] = primary_user
        session['recognized_users'] = recognized_users
        session['all_users'] = ALL_USERS
        return jsonify({
            'status': 'success', 
            'user': primary_user,
            'recognized_users': recognized_users,
            'all_users': ALL_USERS
        })
    else:
        return jsonify({
            'status': 'fail', 
            'recognized_users': recognized_users,
            'all_users': ALL_USERS
        })

@app.route('/capture_face', methods=['POST'])
def capture_face():
    global current_frame
    
    if current_frame is None:
        return jsonify({'status': 'fail', 'message': 'No camera feed available'})
    
    data = request.json
    username = data.get('username')
    
    if not username:
        return jsonify({'status': 'fail', 'message': 'Username required'})
    
    # Capture face for registration
    success, message = capture_face_for_registration(current_frame, username)
    
    if success:
        return jsonify({'status': 'success', 'message': message})
    else:
        return jsonify({'status': 'fail', 'message': message})

@app.route('/train_model', methods=['GET'])
def train_model_route():
    success, message = train_model()
    
    if success:
        return jsonify({'status': 'success', 'message': message})
    else:
        return jsonify({'status': 'fail', 'message': message})

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Please log in first')
        return redirect(url_for('index'))
    
    username = session.get('username', 'User')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Get recognized users and all users from session
    recognized_users = session.get('recognized_users', [])
    all_users = session.get('all_users', [])
    
    return render_template('dashboard.html', 
                          username=username, 
                          now=now, 
                          recognized_users=recognized_users,
                          all_users=all_users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("Starting enhanced face recognition system with registration...")
    print(f"Registered users: {ALL_USERS}")
    app.run(debug=True)