"""
detect_live.py - Optimized Real-Time Sign Language Detection
Features: Two-hand detection, word builder, Hindi mapping, recording, screenshots
Controls: R=Record, S=Screenshot, C=Clear, Q=Quit
OPTIMIZED FOR BETTER FPS
"""
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
from collections import deque, Counter
from datetime import datetime

class SignLanguageDetector:
    def __init__(self, model_path="models/rf_model.joblib"):
        """Initialize detector"""
        os.makedirs("screenshots", exist_ok=True)
        os.makedirs("recordings", exist_ok=True)
        
        print("=" * 70)
        print("🤖 Loading Model...")
        try:
            model_data = joblib.load(model_path)
            self.model = model_data["model"]
            self.label_encoder = model_data["label_encoder"]
            self.feature_size = model_data.get("feature_size", 126)
            print(f"✅ Model loaded successfully")
            print(f"   Classes: {len(self.label_encoder.classes_)}")
            print(f"   Feature size: {self.feature_size}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
        
        self.HINDI_MAP = {
            'A':'क','B':'ख','C':'ग','D':'घ','E':'ङ','F':'च','G':'छ','H':'ज','I':'झ','J':'ञ',
            'K':'ट','L':'ठ','M':'ड','N':'ढ','O':'ण','P':'त','Q':'थ','R':'द','S':'ध','T':'न',
            'U':'प','V':'फ','W':'ब','X':'भ','Y':'म','Z':'य'
        }
        
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        self.buffer = deque(maxlen=10)
        self.recording = False
        self.video_writer = None
        self.prev_time = time.time()
        self.fps_history = deque(maxlen=30)
        
        print("=" * 70 + "\n")
    
    def extract_two_hands(self, results):
        """Extract features from both hands (126 features) - SAME AS ORIGINAL"""
        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_label = handedness.classification[0].label
                
                # Faster extraction using list comprehension
                landmarks = [coord for lm in hand_landmarks.landmark 
                           for coord in (lm.x, lm.y, lm.z)]
                
                landmarks_array = np.array(landmarks, dtype=np.float32)
                
                if hand_label == "Left":
                    left_hand = landmarks_array
                elif hand_label == "Right":
                    right_hand = landmarks_array
        
        return np.concatenate([left_hand, right_hand])
    
    def calculate_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        return int(fps), int(avg_fps)
    
    def draw_overlay(self, frame, prediction_data):
        """Draw optimized overlay"""
        h, w = frame.shape[:2]
        
        letter = prediction_data['letter']
        hindi = prediction_data['hindi']
        confidence = prediction_data['confidence']
        hand_detected = prediction_data['hand_detected']
        num_hands = prediction_data['num_hands']
        fps = prediction_data['fps']
        avg_fps = prediction_data['avg_fps']
        
        # Top bar - simplified
        cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 40), -1)
        
        fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps} | Avg: {avg_fps}", 
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        hand_status = f"Hands: {num_hands}/2"
        hand_color = (0, 255, 0) if num_hands == 2 else (0, 165, 255) if num_hands == 1 else (255, 100, 100)
        cv2.putText(frame, hand_status, (w - 140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        
        # Prediction display
        if hand_detected:
            cv2.rectangle(frame, (0, h - 80), (400, h), (20, 20, 40), -1)
            
            pred_text = f"{letter} - {hindi}"
            conf_color = (0, 255, 0) if confidence > 80 else (0, 255, 255) if confidence > 60 else (0, 165, 255)
            
            cv2.putText(frame, pred_text, (10, h - 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, conf_color, 2)
            cv2.putText(frame, f"Conf: {confidence:.1f}%", (10, h - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (w - 30, 25), 8, (0, 0, 255), -1)
        
        return frame
    
    def start_recording(self, frame_size):
        """Start recording"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/demo_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, frame_size)
        self.recording = True
        print(f"🎥 Recording: {filename}")
        return filename
    
    def stop_recording(self):
        """Stop recording"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        print("⏹️  Recording stopped")
    
    def save_screenshot(self, frame):
        """Save screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot: {filename}")
        return filename
    
    def run(self):
        """Main detection loop - FIXED"""
        cap = cv2.VideoCapture(0)
        # REDUCED RESOLUTION FOR BETTER FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ ERROR: Cannot open camera")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("🎥 Starting Detection (Optimized)")
        print("=" * 70)
        print("Controls: R=Record | S=Screenshot | Q=Quit")
        print("=" * 70 + "\n")
        
        # OPTIMIZED MEDIAPIPE SETTINGS
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # LITE MODEL for better FPS
        ) as hands:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                fps, avg_fps = self.calculate_fps()
                
                hand_detected = False
                num_hands = 0
                letter = "?"
                confidence = 0.0
                hindi = ""
                
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                hand_detected = bool(results.multi_hand_landmarks)
                num_hands = len(results.multi_hand_landmarks) if hand_detected else 0
                
                # Draw landmarks - simplified for speed
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                color=(0, 255, 0), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                color=(255, 255, 255), thickness=1)
                        )
                
                # IMPROVED: Better prediction handling
                if hand_detected:
                    features = self.extract_two_hands(results).reshape(1, -1)
                    probs = self.model.predict_proba(features)[0]
                    idx = np.argmax(probs)
                    letter = str(self.label_encoder.inverse_transform([idx])[0]).upper()
                    confidence = probs[idx] * 100
                    hindi = self.HINDI_MAP.get(letter, "")
                    
                    self.buffer.append(letter)
                    final_letter = Counter(self.buffer).most_common(1)[0][0]
                    letter = final_letter
                else:
                    self.buffer.clear()
                
                prediction_data = {
                    'letter': letter, 'hindi': hindi, 'confidence': confidence,
                    'hand_detected': hand_detected, 'num_hands': num_hands,
                    'fps': fps, 'avg_fps': avg_fps
                }
                
                frame = self.draw_overlay(frame, prediction_data)
                
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)
                
                cv2.imshow("SignSync - Two Hand Detection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n✅ Exiting...")
                    break
                elif key == ord('r'):
                    if not self.recording:
                        self.start_recording((frame_width, frame_height))
                    else:
                        self.stop_recording()
                elif key == ord('s'):
                    self.save_screenshot(frame)
        
        if self.recording:
            self.stop_recording()
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.fps_history:
            print(f"\n📊 Average FPS: {sum(self.fps_history) / len(self.fps_history):.2f}")
        print("✅ Program terminated")

def main():
    try:
        detector = SignLanguageDetector()
        detector.run()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()