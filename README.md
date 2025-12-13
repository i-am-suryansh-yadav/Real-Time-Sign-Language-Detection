# Real-Time-Sign-Language-Detection
# Real-Time Indian Sign Language (ISL) A–Z Detection  
### Developed by **Suryansh Yadav**

A real-time computer vision system that detects **Indian Sign Language alphabet gestures (A–Z)** using MediaPipe, OpenCV, and a machine-learning model trained on 63-dimensional hand landmark features.

The system performs:
- Live webcam hand landmark detection  
- Real-time prediction of ISL alphabet  
- Hindi letter mapping (क, ख, ग, ...)  
- Confidence percentage display  
- Word formation from detected letters  
- Optional demo recording  
- Web-based live streaming through Flask  

---

## 🚀 Features

### 🎥 **Real-Time Hand Tracking**
- Uses **MediaPipe Hands** to extract 21 landmark points (63 values).
- Displays landmarks on webcam feed.

### 🔤 **ISL Alphabet Recognition**
- Trained ML model (RandomForest) achieves **98–100% accuracy** with clean dataset.
- Predicts letter + Hindi equivalent:
A → क
B → ख
C → ग
...

### 🔍 **Confidence Percentage**
- Shows how confident the model is (e.g., 99.2%).

### 📝 **Word Builder**
Forms words from sequential predictions:
H E L L O → HELLO

### 💾 **Demo Recording**
Press **R** to record a full demo video automatically:
demo_week1.mp4

### 🌐 **Web App (Flask)**
- Live streaming in browser using MJPEG feed.
- Works on Chrome, Edge, Firefox.

## 📁 Project Structure

Real-Time-Sign-Language-Detection/
│
├── data/
│ └── isl_landmarks.csv # training dataset (ignored in git)
│
├── models/
│ └── rf_model.joblib # trained model (ignored in git)
│
├── screenshots/ # auto-saved screenshots
│
├── src/
│ ├── hand_landmarks.py # view 63 landmark points
│ ├── train_model.py # train ML classifier
│ ├── detect_live.py # run live ISL detection
│ └── web_app.py # Flask app for browser streaming
│
├── ui/
│ ├── index.html # frontend landing page
│ └── style.css # neon-themed UI
│
├── .gitignore
├── README.md
└── demo_week1.mp4 (ignored in git)

## 🛠 Requirements
Install all dependencies inside the virtual environment:

pip install opencv-python mediapipe scikit-learn numpy pandas joblib flask
⚙️ Setup Guide (Step-by-Step)
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/Real-Time-Sign-Language-Detection.git

cd Real-Time-Sign-Language-Detection

2️⃣ Create and Activate Virtual Environment
python -m venv env
env\Scripts\activate

3️⃣ Install Libraries
pip install -r requirements.txt
(or install manually)

4️⃣ Add Dataset
Place your CSV in:data/isl_landmarks.csv

5️⃣ Train the Model
python src/train_model.py

▶️ Running the Real-Time Detector
Start webcam prediction app:python src/detect_live.py
Keyboard Controls:
Key	Action
r	Start/stop recording demo video
s	Save screenshot
c	Clear formed word
q	Quit

🌐 Running the Web Application (Browser View)
Start server:python src/web_app.py
Open browser:http://localhost:500

You will see:
Live webcam
Landmarks
Prediction, Hindi letter, confidence
Credits

📸 Screenshots
Screenshots are saved automatically in:screenshots/

Recommended screenshots for documentation:
Each letter A–Z
System UI
Hindi mapping
Word formation example
Web app interface
Recording in progress

🧠 Technologies Used
Technology	Purpose
MediaPipe Hands	Real-time 21-point hand landmark extraction
OpenCV	Webcam capture, drawing, display
Scikit-learn	Training RandomForest classifier
NumPy	Vector operations for model input
Joblib	Saving/loading ML model
Flask	Web-app live streaming
HTML/CSS	Neon-themed UI

🎯 Accuracy Achieved
Dataset: 26 letters × 50 samples each = 1300 rows
Average accuracy: 98–100% with clean data
Smooth real-time performance at 30 FPS

👨‍💻 Developer
Developed by:Suryansh Yadav