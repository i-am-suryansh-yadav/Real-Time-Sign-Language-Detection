import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, "Press SPACE to print 63 numbers. Press q to quit.", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Hand Landmarks", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == 32:  # space bar
                if results.multi_hand_landmarks:
                    # take first hand
                    lms = results.multi_hand_landmarks[0].landmark
                    flat = []
                    for lm in lms:
                        flat.extend([lm.x, lm.y, lm.z])
                    # Print as one line (63 numbers)
                    print(" ".join(map(str, flat)))
                else:
                    print("No hand detected – try again.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()