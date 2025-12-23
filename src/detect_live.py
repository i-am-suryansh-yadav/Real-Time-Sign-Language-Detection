# detect_live.py (With recording toggle on 'r', Hindi mapping, word builder, improved overlay for beautiful UI, optimized FPS)
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
from collections import deque, Counter
from datetime import datetime

os.makedirs("screenshots", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

# Load model
data = joblib.load("models/rf_model.joblib")
model = data["model"]
le = data["label_encoder"]

# Hindi map
HINDI_MAP = {
    'A':'क','B':'ख','C':'ग','D':'घ','E':'ङ','F':'च','G':'छ','H':'ज','I':'झ','J':'ञ',
    'K':'ट','L':'ठ','M':'ड','N':'ढ','O':'ण','P':'त','Q':'थ','R':'द','S':'ध','T':'न',
    'U':'प','V':'फ','W':'ब','X':'भ','Y':'म','Z':'य'
}

# Globals
buffer = deque(maxlen=10)
spoken = ""
word_builder = []
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# Optimize camera for FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_time = time.time()

recording = False
video_writer = None

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

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        # No resize here - already set to 640x480 for faster processing

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = bool(results.multi_hand_landmarks)

        for h in results.multi_hand_landmarks or []:
            mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        features = extract_two_hands(results).reshape(1, -1)
        probs = model.predict_proba(features)[0]
        idx = np.argmax(probs)

        pred = str(le.inverse_transform([idx])[0]).upper()
        conf = probs[idx] * 100

        buffer.append(pred)
        final = Counter(buffer).most_common(1)[0][0]

        # Word builder
        if final != spoken:
            spoken = final
            word_builder.append(final)
            if len(word_builder) > 12:
                word_builder = word_builder[-12:]

        fps = int(1 / (time.time() - prev_time + 1e-5))
        prev_time = time.time()

        # Beautiful overlay inspired by YT: semi-transparent black box for status, big prediction text with shadow
        overlay = frame.copy()
        
        # Status box (top left)
        cv2.rectangle(overlay, (10, 10), (300, 40), (0, 0, 0), -1)
        cv2.putText(overlay, f"FPS: {fps} | Hand: {'Detected' if hand_detected else 'Not Detected'}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Prediction box (bottom left, larger)
        text = f"{final} - {HINDI_MAP.get(final, '')} ({conf:.1f}%)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.rectangle(overlay, (10, frame.shape[0] - th - 40), (10 + tw + 20, frame.shape[0] - 20), (0, 0, 0), -1)
        # Shadow text
        cv2.putText(overlay, text, (15 + 2, frame.shape[0] - 25 + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        # Main text
        cv2.putText(overlay, text, (15, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Word builder
        word_text = "Word: " + ''.join(word_builder)
        cv2.putText(overlay, word_text, (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Apply semi-transparent overlay
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        if recording:
            cv2.putText(frame, "REC", (frame.shape[1] - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            video_writer.write(frame)

        cv2.imshow("Two-Hand ISL Detector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            if not recording:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"recordings/demo_{ts}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(path, fourcc, 30, (frame.shape[1], frame.shape[0]))
                recording = True
            else:
                recording = False
                video_writer.release()
                video_writer = None

        if key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshots/screenshot_{ts}.jpg", frame)

        if key == ord('c'):  # Clear word builder
            word_builder = []

        if key == ord('q'):
            break

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()