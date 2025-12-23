# web_app.py (Fixed: Lowered MediaPipe thresholds to 0.3 for better detection, added model_complexity=0 for faster/lighter processing and higher FPS, removed confidence threshold for appending to restore previous behavior, always display prediction if hand detected even if low conf, improved FPS calculation)
from flask import Flask, Response, render_template, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque, Counter

app = Flask(
    __name__,
    template_folder="../ui",
    static_folder="../ui/static",
    static_url_path="/static"
)

# ================= LOAD MODEL =================
data = joblib.load("models/rf_model.joblib")
model = data["model"]
label_encoder = data["label_encoder"]

# ================= HINDI MAP =================
HINDI_MAP = {
    'A':'क','B':'ख','C':'ग','D':'घ','E':'ङ','F':'च','G':'छ','H':'ज','I':'झ','J':'ञ',
    'K':'ट','L':'ठ','M':'ड','N':'ढ','O':'ण','P':'त','Q':'थ','R':'द','S':'ध','T':'न',
    'U':'प','V':'फ','W':'ब','X':'भ','Y':'म','Z':'य'
}

# ================= GLOBALS =================
cap = None
running = False
detecting = False  # New flag for detection toggle
buffer = deque(maxlen=10)
prev_time = time.time()
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ================= SPELLING MODE ONLY =================
spoken = ""
word_builder = []

# ================= FEATURE EXTRACTION =================
def extract_two_hands(results):
    left = np.zeros(63)
    right = np.zeros(63)
    if results.multi_hand_landmarks:
        for i, hand in enumerate(results.multi_hand_landmarks[:2]):
            pts = []
            for lm in hand.landmark:
                pts.extend([lm.x, lm.y, lm.z])
            if i == 0:
                left = np.array(pts)
            else:
                right = np.array(pts)
    return np.concatenate([left, right])

# ================= VIDEO GENERATOR =================
def gen_frames():
    global prev_time, spoken, word_builder, detecting
    with mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.3,  # Lowered for better detection
        min_tracking_confidence=0.3,   # Lowered for better detection
        model_complexity=0             # Lighter model for faster FPS
    ) as hands:
        while running:
            if cap is None or not cap.isOpened():
                break
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            # No resize - cap already set to 640x480 for FPS
            
            hand_detected = False
            final_letter = "?"
            confidence = 0.0
            hindi = ""
            
            if detecting:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                hand_detected = bool(results.multi_hand_landmarks)
                if hand_detected:
                    for h in results.multi_hand_landmarks or []:
                        mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
                    
                    features = extract_two_hands(results).reshape(1, -1)
                    probs = model.predict_proba(features)[0]
                    idx = np.argmax(probs)
                    pred_letter = str(label_encoder.inverse_transform([idx])[0]).upper()
                    confidence = probs[idx] * 100
                    
                    buffer.append(pred_letter)  # Always append to restore previous behavior
                    final_letter = Counter(buffer).most_common(1)[0][0]
                    hindi = HINDI_MAP.get(final_letter, "")
                    
                    # SPELLING MODE (NO VOICE)
                    if final_letter != spoken:
                        spoken = final_letter
                        word_builder.append(final_letter)
                        if len(word_builder) > 12:
                            word_builder = word_builder[-12:]
                else:
                    final_letter = "No hand"
            else:
                # Raw feed - no processing
                pass
            
            # FPS (improved calculation)
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time + 1e-6)) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # ---------- OVERLAY TEXT (Only if detecting) ----------
            if detecting:
                # Status
                status_text = f"FPS: {fps} | Hand: {'Detected' if hand_detected else 'Not Detected'}"
                cv2.putText(frame, status_text, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Shadow
                cv2.putText(frame, status_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)  # Main
                
                # Prediction (always show if hand detected, with conf)
                if hand_detected:
                    conf_color = (0, 255, 0) if confidence > 70 else (0, 255, 255)  # Green if high, yellow if low
                    text = f"{final_letter} - {hindi} ({confidence:.1f}%)"
                else:
                    text = final_letter
                    conf_color = (0, 0, 255)
                cv2.putText(frame, text, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)  # Shadow
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, conf_color, 2)  # Main
                
                # Word display
                word_text = "Word: " + ''.join(word_builder)
                cv2.putText(frame, word_text, (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)  # Shadow
                cv2.putText(frame, word_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)  # Main
            
            # Encode frame
            _, buffer_img = cv2.imencode(".jpg", frame)
            frame_bytes = buffer_img.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start_camera():
    global cap, running, detecting
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    running = True
    detecting = False  # Start with detection off
    return jsonify({"status": "camera started"})

@app.route("/toggle_detection", methods=["POST"])
def toggle_detection():
    global detecting
    detecting = not detecting
    return jsonify({"detecting": detecting})

@app.route("/stop", methods=["POST"])
def stop_camera():
    global running, detecting, cap
    running = False
    detecting = False
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None
    buffer.clear()  # Clear buffer on stop
    word_builder.clear()
    return jsonify({"status": "camera stopped"})

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)