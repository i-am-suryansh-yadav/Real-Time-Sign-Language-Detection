# ✋ Real-Time Indian Sign Language (ISL) A–Z Detection  
### Developed by **Suryansh Yadav**

A real-time computer vision system that detects **Indian Sign Language alphabet gestures (A–Z)** using MediaPipe, OpenCV, and a machine-learning model trained on 63-dimensional hand landmark features.

---

## 🚀 Features

### 🎥 Real-Time Hand Tracking
- Uses **MediaPipe Hands** to extract 21 landmark points (63 values per hand)
- Displays landmarks on webcam feed in real-time
- Optimized for higher FPS with lower resolution (640x480)

### 🔤 ISL Alphabet Recognition
- Trained ML model (RandomForest) achieves **99% accuracy**
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
- Press **R** to start/stop recording live demo videos (saved as MP4)

### 🌐 Web App (Flask)
- Stream webcam feed + predictions directly in browser
- Works on Chrome, Edge, Firefox
- Beautiful overlay with shadows for text readability

### 📈 FPS Monitoring
- Real-time FPS display to ensure smooth performance

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
│ └── rf_model.joblib # Trained ML model (ignored in Git)  
│  
├── screenshots/ # Saved screenshots (optional)  
│  
├── recordings/ # Recorded demo videos (optional)  
│  
├── src/  
│ ├── convert_images_to_landmarks.py # Image to landmark converter  
│ ├── detect_live.py # Live webcam letter detection with recording  
│ ├── hand_landmarks.py # Hand landmark extraction script  
│ ├── merge_csvs.py # Merge CSVs into one dataset  
│ ├── test_camera.py # Simple camera test script  
│ ├── train_model.py # Train the RandomForest model  
│ └── web_app.py # Flask app for web UI  
│  
├── ui/  
│ ├── index.html # Landing page with features, demo, about  
│ └── static/  
│     ├── hand.png # Hero image  
│     ├── script.js # JS for camera controls  
│     ├── style.css # Styling  
│     └── demo.mp4 # Placeholder for demo video (add your own)  
│  
├── .gitignore  
├── README.md  
└── requirements.txt  

## 🛠 Requirements  

Install all dependencies inside the virtual environment:  

```bash
pip install opencv-python mediapipe scikit-learn numpy pandas joblib flask