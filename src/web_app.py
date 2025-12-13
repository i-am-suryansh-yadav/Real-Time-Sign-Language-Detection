from flask import Flask, Response, render_template_string
import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

app = Flask(__name__)

MODEL_PATH = "models/rf_model.joblib"
HINDI_MAP = {
    'A':'क','B':'ख','C':'ग','D':'घ','E':'ङ','F':'च','G':'छ','H':'ज','I':'झ','J':'ञ',
    'K':'ट','L':'ठ','M':'ड','N':'ढ','O':'ण','P':'त','Q':'थ','R':'द','S':'ध','T':'न',
    'U':'प','V':'फ','W':'ब','X':'भ','Y':'म','Z':'य'
}

# Load model
data = joblib.load(MODEL_PATH)
model = data["model"]
label_encoder = data["label_encoder"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)


def gen():
    last_pred = None
    last_time = 0

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        pred_letter = None
        conf = 0.0

        # Hand detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            flat = []
            for lm in hand_landmarks.landmark:
                flat.extend([lm.x, lm.y, lm.z])

            feat = np.array(flat).reshape(1, -1)

            try:
                proba = model.predict_proba(feat)[0]
                idx = np.argmax(proba)
                pred_letter = label_encoder.inverse_transform([idx])[0].upper()
                conf = proba[idx]
            except:
                pred_letter = None
                conf = 0.0

            # Debounce
            cur_time = time.time()
            if pred_letter != last_pred or (cur_time - last_time) > 0.8:
                last_pred = pred_letter
                last_time = cur_time

        # Draw prediction (NO BOX)
        if pred_letter:
            text = f"{pred_letter} - {HINDI_MAP.get(pred_letter,'')} ({conf*100:.1f}%)"
            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        else:
            cv2.putText(frame, "Show your hand clearly",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

        # Developer credit
        cv2.putText(frame, "Developed by suryansh Yadav",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 2)

        # Encode frame to JPEG
        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    # Simple streaming page
    html = """
    <html>
    <head>
        <title>ISL Live Detector</title>
        <style>
            body {
                background:#060b1b;
                color:white;
                text-align:center;
                font-family:Arial;
            }
            h1 { margin-top:20px; }
        </style>
    </head>
    <body>
        <h1>Real-Time ISL A–Z Detector</h1>
        <p>Live Stream from Webcam</p>
        <img src="/video_feed" width="900" />
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)