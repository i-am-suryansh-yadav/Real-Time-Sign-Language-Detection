"""
web_app.py - OPTIMIZED Flask Web Application
Real-time sign language detection without word formation
OPTIMIZED FOR PERFORMANCE
"""
from flask import Flask, Response, render_template, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import os
from collections import deque, Counter

# ==================== PATH CONFIGURATION ====================
# Get the absolute path of the current file's directory (src folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to project root
PROJECT_ROOT = os.path.dirname(BASE_DIR)

UI_PATH = os.path.join(PROJECT_ROOT, "ui")
STATIC_PATH = os.path.join(UI_PATH, "static")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_model.joblib")

# Print paths for debugging
print("=" * 70)
print("📂 Path Configuration:")
print(f"BASE_DIR (src): {BASE_DIR}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"UI_PATH: {UI_PATH}")
print(f"STATIC_PATH: {STATIC_PATH}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"\n📁 File Check:")
print(f"UI folder exists: {os.path.exists(UI_PATH)}")
print(f"Static folder exists: {os.path.exists(STATIC_PATH)}")
print(f"demo.mp4 exists: {os.path.exists(os.path.join(STATIC_PATH, 'demo.mp4'))}")
print(f"Model file exists: {os.path.exists(MODEL_PATH)}")

app = Flask(__name__, 
            template_folder=UI_PATH, 
            static_folder=STATIC_PATH, 
            static_url_path="/static")

# ==================== CONFIG ====================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ==================== LOAD MODEL ====================
print("=" * 70)
print("🤖 Loading Model...")
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    label_encoder = model_data["label_encoder"]
    print(f"✅ Model loaded | Classes: {len(label_encoder.classes_)}")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    print(f"❌ Expected model at: {MODEL_PATH}")
    print(f"\n💡 Please ensure 'rf_model.joblib' exists in the 'models' folder")
    raise

HINDI_MAP = {
    'A':'क','B':'ख','C':'ग','D':'घ','E':'ङ','F':'च','G':'छ','H':'ज','I':'झ','J':'ञ',
    'K':'ट','L':'ठ','M':'ड','N':'ढ','O':'ण','P':'त','Q':'थ','R':'द','S':'ध','T':'न',
    'U':'प','V':'फ','W':'ब','X':'भ','Y':'म','Z':'य'
}

# ==================== GLOBAL STATE ====================
class AppState:
    def __init__(self):
        self.cap = None
        self.running = False
        self.detecting = False
        self.buffer = deque(maxlen=8)
        self.prev_time = time.time()
        self.fps_history = deque(maxlen=20)
        self.lock = threading.Lock()
        
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

state = AppState()
print("=" * 70 + "\n")

# ==================== HELPER FUNCTIONS ====================
def extract_two_hands(results):
    """SAME AS ORIGINAL: Fast feature extraction"""
    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            landmarks_array = np.array(landmarks, dtype=np.float32)
            
            if hand_label == "Left":
                left_hand = landmarks_array
            elif hand_label == "Right":
                right_hand = landmarks_array
    
    return np.concatenate([left_hand, right_hand])

def calculate_fps():
    """Calculate FPS"""
    current_time = time.time()
    fps = 1 / (current_time - state.prev_time + 1e-6)
    state.prev_time = current_time
    state.fps_history.append(fps)
    return int(fps), int(sum(state.fps_history) / len(state.fps_history))

# ==================== VIDEO GENERATOR ====================
def generate_frames():
    """Video generation - NO FRAME SKIPPING"""
    with state.mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0  # LITE MODEL
    ) as hands:
        
        while state.running:
            success, frame = state.cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            fps, avg_fps = calculate_fps()
            
            hand_detected = False
            num_hands = 0
            letter = "?"
            confidence = 0.0
            hindi = ""
            
            if state.detecting:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                hand_detected = bool(results.multi_hand_landmarks)
                num_hands = len(results.multi_hand_landmarks) if hand_detected else 0
                
                # Simplified landmark drawing for speed
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        state.mp_draw.draw_landmarks(
                            frame, hand_landmarks, state.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                color=(0, 255, 0), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                color=(255, 255, 255), thickness=1)
                        )
                
                # Prediction
                if hand_detected:
                    features = extract_two_hands(results).reshape(1, -1)
                    probs = model.predict_proba(features)[0]
                    idx = np.argmax(probs)
                    pred_letter = str(label_encoder.inverse_transform([idx])[0]).upper()
                    confidence = probs[idx] * 100
                    
                    state.buffer.append(pred_letter)
                    letter = Counter(state.buffer).most_common(1)[0][0]
                    hindi = HINDI_MAP.get(letter, "")
            
            # SIMPLIFIED OVERLAY
            # Top bar
            cv2.rectangle(frame, (0, 0), (w, 40), (10, 10, 30), -1)
            
            status_text = f"FPS: {fps} | Hands: {num_hands}/2"
            status_color = (0, 255, 0) if state.detecting else (100, 100, 100)
            cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            detect_status = "DETECTING" if state.detecting else "PAUSED"
            detect_color = (0, 255, 0) if state.detecting else (255, 100, 100)
            cv2.putText(frame, detect_status, (w - 130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, detect_color, 2)
            
            # Prediction display
            if state.detecting and hand_detected:
                cv2.rectangle(frame, (0, h - 70), (w, h), (10, 10, 30), -1)
                
                pred_text = f"{letter} - {hindi}"
                conf_color = (0, 255, 0) if confidence > 80 else (0, 255, 255) if confidence > 60 else (100, 150, 255)
                
                cv2.putText(frame, pred_text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, conf_color, 2)
                cv2.putText(frame, f"{confidence:.1f}%", (10, h - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==================== ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html")

# Add range request support for video streaming
@app.route('/static/<path:filename>')
def serve_static_with_range(filename):
    """Serve static files with range request support for video streaming"""
    from flask import send_from_directory, request, make_response
    import os
    
    file_path = os.path.join(app.static_folder, filename)
    
    if not os.path.exists(file_path):
        return "File not found", 404
    
    # Check if this is a video file
    if filename.endswith(('.mp4', '.webm', '.ogg')):
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Check for Range header
        range_header = request.headers.get('Range')
        
        if range_header:
            # Parse range header
            byte_range = range_header.replace('bytes=', '').split('-')
            start = int(byte_range[0]) if byte_range[0] else 0
            end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1
            
            # Read the requested chunk
            with open(file_path, 'rb') as f:
                f.seek(start)
                data = f.read(end - start + 1)
            
            # Create response with partial content
            response = make_response(data)
            response.status_code = 206  # Partial Content
            response.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
            response.headers['Content-Length'] = str(len(data))
            response.headers['Content-Type'] = 'video/mp4'
            response.headers['Accept-Ranges'] = 'bytes'
            return response
    
    # For non-video files or if no range requested, serve normally
    return send_from_directory(app.static_folder, filename)

@app.route("/start", methods=["POST"])
def start_camera():
    """Start camera"""
    with state.lock:
        try:
            if state.cap is None or not state.cap.isOpened():
                state.cap = cv2.VideoCapture(0)
                state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                state.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Warm up camera
                for _ in range(3):
                    state.cap.read()
            
            state.running = True
            state.detecting = False
            print("✅ Camera started")
            return jsonify({"status": "camera started"})
        except Exception as e:
            print(f"❌ Camera error: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/toggle_detection", methods=["POST"])
def toggle_detection():
    with state.lock:
        state.detecting = not state.detecting
        
        if not state.detecting:
            state.buffer.clear()
        
        print(f"🔄 Detection {'ON' if state.detecting else 'OFF'}")
    
    return jsonify({"detecting": state.detecting})

@app.route("/stop", methods=["POST"])
def stop_camera():
    with state.lock:
        state.running = False
        state.detecting = False
        
        if state.cap is not None and state.cap.isOpened():
            state.cap.release()
            state.cap = None
        
        state.buffer.clear()
        state.fps_history.clear()
        
        print("⏹️ Camera stopped")
    
    return jsonify({"status": "camera stopped"})

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/status")
def status():
    with state.lock:
        return jsonify({
            "running": state.running,
            "detecting": state.detecting,
            "fps": int(sum(state.fps_history) / len(state.fps_history)) if state.fps_history else 0
        })

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ==================== CLEANUP ====================
def cleanup():
    with state.lock:
        if state.cap is not None:
            state.cap.release()
            state.cap = None
    print("\n🧹 Cleanup done")

import atexit
atexit.register(cleanup)

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("SignSync - OPTIMIZED")
    print("🚀" * 35)
    print(f"\n✅ Ready: http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)