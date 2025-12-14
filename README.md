# ✋ Real-Time Indian Sign Language (ISL) A–Z Detection  
### Developed by **Suryansh Yadav**

A real-time computer vision system that detects **Indian Sign Language alphabet gestures (A–Z)** using MediaPipe, OpenCV, and a machine-learning model trained on 63-dimensional hand landmark features.

---

## 🚀 Features

### 🎥 Real-Time Hand Tracking
- Uses **MediaPipe Hands** to extract 21 landmark points (63 values)
- Displays landmarks on webcam feed in real-time

### 🔤 ISL Alphabet Recognition
- Trained ML model (RandomForest) achieves **98–100% accuracy**
- Predicts letter + Hindi equivalent:
  - A → क
  - B → ख
  - C → ग
  - ...

### 🎯 Confidence Percentage
- Displays model confidence (e.g., `99.2%`) next to predictions

### 🧩 Word Builder
- Forms words from sequential predictions  
  `H E L L O → HELLO`

### 💾 Demo Recording
- Press **R** to record live demo: `demo_week1.mp4`

### 🌐 Web App (Flask)
- Stream webcam feed + predictions directly in browser
- Works on Chrome, Edge, Firefox

---

## 📊 Dataset Source

This project uses publicly available datasets from Kaggle:

- 🔗 [Indian Sign Language Dataset by Soumya Kushwaha](https://www.kaggle.com/datasets/soumyakushwaha/indian-sign-language-dataset)
- 🔗 [Indian Sign Language ISL Dataset by Prathuma Rikeri](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)

These contain labeled hand gesture images for A–Z, which are converted to hand landmarks and used for model training.

---

## 📁 Project Structure

Real-Time-Sign-Language-Detection/
│
├── data/
│ └── isl_landmarks.csv # Final merged dataset (ignored in Git)
│
├── models/
│ └── isl_model.pkl # Trained ML model (ignored in Git)
│
├── screenshots/ # Saved screenshots (optional)
│
├── src/
│ ├── convert_photo_to_landmarks.py # Image to landmark converter
│ ├── merge.py # Merge CSVs into one dataset
│ ├── train_model.py # Train the RandomForest model
│ ├── detect_live.py # Live webcam letter detection
│ └── web_app.py # Flask app for web UI
│
├── ui/
│ ├── index.html # Landing page
│ ├── style.css # Neon-themed styling
│ └── static/ # Static resources like CSS
│
├── .gitignore
├── README.md

## 🛠 Requirements

Install all dependencies inside the virtual environment:

pip install opencv-python mediapipe scikit-learn numpy pandas joblib flask

⚙️ Setup Guide
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/Real-Time-Sign-Language-Detection.git
cd Real-Time-Sign-Language-Detection
2️⃣ Create and Activate Virtual Environment
python -m venv env
env\Scripts\activate
3️⃣ Install Libraries
pip install -r requirements.txt
# or install manually as shown above
4️⃣ Add Dataset
Place your final CSV here:
data/isl_landmarks.csv
5️⃣ Train the Model
python src/train_model.py
▶️ Run the Real-Time Detector
python src/detect_live.py

🎛 Keyboard Controls:
Key	Action
r	Start/stop recording
s	Save screenshot
c	Clear formed word
q	Quit

🌐 Web Application (Browser View)
python src/web_app.py
Then open:
http://localhost:5000

You will see:
Live webcam
Detected landmarks
Prediction (A–Z)
Hindi letter
Confidence score

📸 Screenshots
Screenshots are saved in the screenshots/ folder. Recommended shots:

Each letter A–Z

Word builder in action

Hindi letter display

Flask web app interface

🧠 Technologies Used
Technology	Purpose
MediaPipe Hands	Real-time hand landmark extraction
OpenCV	Webcam video + annotation overlay
scikit-learn	ML classification (RandomForest)
NumPy	Vector preprocessing
Joblib	Save/load ML models
Flask	Live browser-based UI
HTML/CSS	Neon-style user interface

🎯 Accuracy Achieved
Dataset: 26 letters × 50 samples × 5 datasets = 6,500+ samples

Accuracy: 98–100% on clean data

Inference speed: Real-time (30 FPS on average webcam)

👨‍💻 Developer
Developed by:
Suryansh Yadav
December 2025