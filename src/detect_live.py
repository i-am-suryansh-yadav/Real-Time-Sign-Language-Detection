import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
import os

MODEL_PATH = "models/rf_model.joblib"
HINDI_MAP = {'A':'क', 'B':'ख', 'C':'ग', 'D':'घ', 'E':'ङ', 'F':'च', 'G':'छ',
             'H':'ज', 'I':'झ', 'J':'ञ', 'K':'ट', 'L':'ठ', 'M':'ड', 'N':'ढ',
             'O':'ण', 'P':'त', 'Q':'थ', 'R':'द', 'S':'ध', 'T':'न', 'U':'प',
             'V':'फ', 'W':'ब', 'X':'भ', 'Y':'म', 'Z':'य'}

# Load model + encoder
data = joblib.load(MODEL_PATH)
model = data["model"]
label_encoder = data["label_encoder"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_landmarks_list(hand_landmarks):
    flat = []
    for lm in hand_landmarks.landmark:
        flat.extend([lm.x, lm.y, lm.z])
    return flat


def main(display_width=1280, display_height=720):

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

    # Recording variables
    writer = None
    recording = False
    output_filename = "demo_week1.mp4"

    last_pred = None
    last_time = 0
    current_word = ""

    screenshots_dir = "screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)

    window_name = "Real-Time ISL A-Z Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            original_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            pred_letter = None
            conf = 0.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                feat = np.array(get_landmarks_list(hand_landmarks)).reshape(1, -1)

                try:
                    proba = model.predict_proba(feat)[0]
                    idx = np.argmax(proba)
                    pred_label = label_encoder.inverse_transform([idx])[0]
                    pred_letter = pred_label.upper()
                    conf = proba[idx]
                except:
                    pred_letter = None
                    conf = 0.0

                cur_time = time.time()
                if pred_letter and (pred_letter != last_pred or (cur_time - last_time) > 0.8):
                    last_pred = pred_letter
                    last_time = cur_time
                    if conf > 0.45:
                        current_word += pred_letter

            # Show prediction
            if pred_letter:
                text = f"{pred_letter} - {HINDI_MAP.get(pred_letter,'')} ({conf*100:.1f}%)"
                cv2.putText(frame, text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            else:
                cv2.putText(frame,
                            "Show your hand - 's' screenshot | 'c' clear word | 'r' record | 'q' quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

            # Current word
            cv2.putText(frame, f"Word: {current_word}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            # Credits
            cv2.putText(frame, "Developed by suryansh Yadav", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 2)

            # Recording status
            if recording:
                cv2.putText(frame, "● Recording...", (display_width - 280, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

            # Display enlarged frame
            disp_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_name, disp_frame)

            # Write video if recording
            if recording and writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord('q'):
                break

            # Clear word
            if key == ord('c'):
                current_word = ""

            # Save screenshot
            if key == ord('s'):
                fname = os.path.join(screenshots_dir, f"shot_{int(time.time())}.png")
                cv2.imwrite(fname, original_frame)
                print("Saved screenshot:", fname)

            # Toggle recording (start/stop)
            if key == ord('r'):
                if not recording:
                    print("Recording started...")
                    recording = True
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 20.0
                    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_w, frame_h))
                else:
                    print("Recording stopped. Saved:", output_filename)
                    recording = False
                    if writer is not None:
                        writer.release()
                    writer = None

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(display_width=1280, display_height=720)