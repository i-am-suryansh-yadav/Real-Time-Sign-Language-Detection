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
    static_folder="../ui/static"
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
buffer = deque(maxlen=10)
prev_time = time.time()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


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
    global prev_time

    with mp_hands.Hands(max_num_hands=2) as hands:
        while running:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, None, fx=0.7, fy=0.7)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_detected = bool(results.multi_hand_landmarks)

            for h in results.multi_hand_landmarks or []:
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

            features = extract_two_hands(results).reshape(1, -1)
            probs = model.predict_proba(features)[0]
            idx = np.argmax(probs)

            pred_letter = str(label_encoder.inverse_transform([idx])[0]).upper()
            confidence = probs[idx] * 100

            buffer.append(pred_letter)
            final_letter = Counter(buffer).most_common(1)[0][0]

            # FPS
            fps = int(1 / (time.time() - prev_time))
            prev_time = time.time()

            # ---------- OVERLAY TEXT ----------
            cv2.putText(
                frame,
                f"{final_letter} - {HINDI_MAP.get(final_letter,'')} ({confidence:.1f}%)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps} | Hand: {'Detected' if hand_detected else 'Not Detected'}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1
            )

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
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    return jsonify({"status": "camera started"})


@app.route("/stop", methods=["POST"])
def stop_camera():
    global running
    running = False
    if cap:
        cap.release()
    return jsonify({"status": "camera stopped"})


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=False)
