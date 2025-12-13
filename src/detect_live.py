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

data = joblib.load("models/rf_model.joblib")
model = data["model"]
le = data["label_encoder"]

buffer = deque(maxlen=10)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
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

with mp_hands.Hands(max_num_hands=2) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
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

        pred = str(le.inverse_transform([idx])[0]).upper()
        conf = probs[idx]

        buffer.append(pred)
        final = Counter(buffer).most_common(1)[0][0]

        fps = int(1 / (time.time() - prev_time))
        prev_time = time.time()

        # Status
        cv2.putText(frame,
                    f"FPS: {fps} | Hand: {'Detected' if hand_detected else 'Not Detected'}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        # Prediction (NO HINDI HERE)
        text = f"{final} ({conf*100:.1f}%)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (10, 60-th), (10+tw+12, 70), (0,0,255), 2)
        cv2.putText(frame, text, (16, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        if recording:
            cv2.putText(frame, "REC",
                        (frame.shape[1]-60, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
            video_writer.write(frame)

        cv2.imshow("Two-Hand ISL Detector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            if not recording:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"recordings/demo_{ts}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    path, fourcc, 20,
                    (frame.shape[1], frame.shape[0])
                )
                recording = True
            else:
                recording = False
                video_writer.release()

        if key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshots/screenshot_{ts}.jpg", frame)

        if key == ord('q'):
            break

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()