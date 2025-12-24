# 🖐️ SignSync - Real-Time Indian Sign Language Detection

An AI-powered web application for real-time Indian Sign Language (ISL) detection using computer vision and machine learning.

## 🌟 Features

### Core Features
- **Two-Hand Detection**: Detects gestures using one or both hands (126 features)
- **Real-Time Recognition**: A-Z letter recognition with confidence scores
- **Hindi Mapping**: Maps English letters to corresponding Hindi characters for ISL
- **Word Builder**: Constructs words from sequential sign predictions
- **Beautiful UI**: Modern neon-themed cyberpunk interface

### Technical Features
- **High Performance**: Optimized for 30+ FPS on standard hardware
- **Browser-Based**: No installation required for end users
- **Recording & Screenshots**: Save sessions for demonstrations or analysis
- **Prediction Confidence**: Real-time confidence percentage display
- **FPS Monitoring**: Performance tracking and display

## 📁 Project Structure

```
Real-Time-Sign-Language-Detection/
├── data/
│   ├── final_landmarks.csv          # Training dataset (126 features)
│   └── captured_landmarks/          # Auto-generated landmark captures
├── models/
│   └── rf_model.joblib               # Trained Random Forest model
├── ui/
│   ├── index.html                    # Main HTML file (neon theme)
│   └── static/
│       ├── style.css                 # Neon cyberpunk styles
│       ├── script.js                 # Frontend JavaScript
│       └── hand.png                  # Hero image
├── src/
│   ├── test_camera.py                # Camera testing tool
│   ├── hand_landmarks.py             # Landmark detection & data collection
│   ├── train_model.py                # Model training pipeline
│   ├── detect_live.py                # Standalone detection (OpenCV)
│   └── web_app.py                    # Flask web application
├── screenshots/                       # Auto-generated screenshots
├── recordings/                        # Auto-generated video recordings
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Real-Time-Sign-Language-Detection
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv env
   
   # Windows
   env\Scripts\activate
   
   # Linux/Mac
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Option 1: Web Application (Recommended)

```bash
cd src
python web_app.py
```

Then open your browser to: `http://localhost:5000`

**Controls:**
- Click "Launch Camera" to start
- Click "Start Detection" to begin recognizing signs
- Watch the camera border glow when detecting!
- Navigate between pages (Home, Features, Demo, About)

#### Option 2: Standalone Application

```bash
cd src
python detect_live.py
```

**Controls:**
- `R` - Start/Stop recording
- `S` - Take screenshot
- `C` - Clear word builder
- `Q` - Quit

## 🛠️ Development Workflow

### 1. Test Camera
First, verify your camera is working:
```bash
cd src
python test_camera.py
```

**Controls:**
- `Q` - Quit
- `I` - Show camera info
- `S` - Take screenshot

### 2. Collect Data (Optional)
If you want to collect your own training data:
```bash
cd src
python hand_landmarks.py
```

**Controls:**
- `SPACE` - Print 126 landmark values
- `S` - Save landmarks to CSV
- `C` - Capture with label (for dataset)
- `Q` - Quit

### 3. Train Model
Train on your dataset:
```bash
cd src
python train_model.py
```

This will:
- Load data from `data/final_landmarks.csv`
- Train a Random Forest classifier
- Perform cross-validation
- Show detailed metrics
- Save model to `models/rf_model.joblib`

### 4. Run Detection
Use either `detect_live.py` or `web_app.py` (see Quick Start above)

## 📊 Dataset Format

Your `final_landmarks.csv` should have:
- **126 feature columns**: 
  - Columns 0-62: Left hand landmarks (21 points × 3 coords)
  - Columns 63-125: Right hand landmarks (21 points × 3 coords)
- **1 label column**: The sign letter (A-Z)

Example structure:
```csv
L_x0,L_y0,L_z0,...,L_x20,L_y20,L_z20,R_x0,R_y0,R_z0,...,R_x20,R_y20,R_z20,label
0.5,0.6,0.01,...,0.0,0.0,0.0,...,0.7,0.8,0.02,A
...
```

## 🎨 UI Features

### Neon Theme
- Dark cyberpunk background with animated gradients
- Neon cyan and magenta color scheme
- Glowing effects on all interactive elements

### Camera Detection Glow
When you click "Start Detection":
- Camera border glows with neon light
- Pulsing animation indicates active detection
- Corner brackets appear during detection

### Page Navigation
- **Home**: Hero section + camera controls
- **Features**: Feature showcase with hover effects
- **Demo**: Video demonstration section
- **About**: Project info and developer details

## 🔧 Configuration

### Camera Settings (in web_app.py)
```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
```

### Model Settings (in train_model.py)
```python
n_estimators = 300      # Number of trees
max_depth = None        # Unlimited depth
min_samples_split = 2   # Minimum samples to split
```

### Detection Settings (in detect_live.py / web_app.py)
```python
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
model_complexity = 1  # 0=lite, 1=full
```

## 📈 Performance Tips

1. **Lighting**: Ensure good, even lighting for best detection
2. **Background**: Use a plain background to reduce noise
3. **Hand Position**: Keep hands clearly visible to the camera
4. **Distance**: Stay 1-2 feet from the camera
5. **Resolution**: Lower resolution (640x480) gives better FPS

## 🐛 Troubleshooting

### Camera not opening
```bash
# Test your camera
python src/test_camera.py

# Try different camera index
python src/test_camera.py 1
```

### Low FPS
- Close other applications using the camera
- Reduce camera resolution
- Use `model_complexity=0` for lighter model

### Poor detection accuracy
- Ensure good lighting
- Retrain model with more data
- Check hand visibility in frame
- Verify confidence threshold (default: 60%)

### Web page not loading
- Check if Flask is running
- Verify files are in correct folders:
  - `ui/index.html`
  - `ui/static/style.css`
  - `ui/static/script.js`

## 🤝 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 Technical Details

### Technologies Used
- **Backend**: Python, Flask
- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: scikit-learn, Random Forest
- **Frontend**: HTML5, CSS3, JavaScript
- **UI Design**: Neon/Cyberpunk theme

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Input**: 126 features (63 per hand)
- **Output**: 26 classes (A-Z)
- **Features**: 
  - 21 hand landmarks per hand
  - 3 coordinates per landmark (x, y, z)
  - Both hands combined

### Feature Extraction
Each hand provides:
- Wrist position
- Thumb (4 points)
- Index finger (4 points)
- Middle finger (4 points)
- Ring finger (4 points)
- Pinky finger (4 points)

Total: 21 landmarks × 3 coordinates × 2 hands = 126 features

## 📞 Contact

**Developer**: Suryansh Yadav  
**Email**: suryansh.1251010449@vit.edu  
**Institution**: VIT

## 📄 License

This project is open source and available for educational purposes.

## 🎯 Future Enhancements

- [ ] Voice output for recognized letters
- [ ] Sentence formation with grammar
- [ ] Support for more sign languages
- [ ] Mobile app version
- [ ] Real-time translation to speech
- [ ] Word suggestions and autocomplete
- [ ] Multi-user support
- [ ] Cloud deployment

## 🙏 Acknowledgments

- MediaPipe for hand tracking
- OpenCV for computer vision
- scikit-learn for machine learning
- Flask for web framework
- The ISL community

---

**Made with ❤️ for accessible communication**