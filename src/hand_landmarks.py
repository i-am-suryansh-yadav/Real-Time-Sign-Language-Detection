"""
hand_landmarks.py - Two-Hand Landmark Detection Tool
Captures and displays hand landmarks for both hands (126 features total)
Used for data collection and testing
"""
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

class HandLandmarkDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Output directory for saved landmarks
        self.output_dir = "data/captured_landmarks"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def extract_landmarks(self, results):
        """
        Extract landmarks from both hands
        Returns: numpy array of 126 features (63 per hand)
        """
        left_hand = np.zeros(63)
        right_hand = np.zeros(63)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Get hand label (Left or Right)
                hand_label = handedness.classification[0].label
                
                # Extract 21 landmarks × 3 coordinates = 63 features
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks_array = np.array(landmarks)
                
                # Assign to correct hand
                if hand_label == "Left":
                    left_hand = landmarks_array
                elif hand_label == "Right":
                    right_hand = landmarks_array
        
        # Concatenate both hands: [left_63_features, right_63_features]
        combined = np.concatenate([left_hand, right_hand])
        return combined
    
    def draw_info_overlay(self, frame, num_hands, landmarks_captured, fps):
        """Draw informative overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Title
        cv2.putText(frame, "Hand Landmark Detector", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Hand count
        hand_text = f"Hands Detected: {num_hands}/2"
        color = (0, 255, 0) if num_hands == 2 else (0, 165, 255) if num_hands == 1 else (0, 0, 255)
        cv2.putText(frame, hand_text, 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps}", 
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Landmarks captured
        cv2.putText(frame, f"Captured: {landmarks_captured}", 
                    (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Instructions
        instructions = [
            "SPACE - Print landmarks (126 numbers)",
            "S - Save to file",
            "C - Capture for dataset",
            "Q - Quit"
        ]
        
        y_offset = h - 120
        for instruction in instructions:
            cv2.putText(frame, instruction, 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
        
        return frame
    
    def save_landmarks_to_file(self, landmarks, label=None):
        """Save landmarks to CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"landmarks_{timestamp}.csv")
        
        with open(filename, 'w') as f:
            # Write header
            headers = []
            for hand in ['L', 'R']:
                for i in range(21):
                    headers.extend([f'{hand}_x{i}', f'{hand}_y{i}', f'{hand}_z{i}'])
            if label:
                headers.append('label')
            f.write(','.join(headers) + '\n')
            
            # Write data
            data = ','.join(map(str, landmarks))
            if label:
                data += f',{label}'
            f.write(data + '\n')
        
        return filename
    
    def run(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ ERROR: Cannot open camera")
            return
        
        print("=" * 70)
        print("Hand Landmark Detector - Two Hand Support (126 features)")
        print("=" * 70)
        print("Controls:")
        print("  SPACE - Print 126 landmark values to console")
        print("  S     - Save landmarks to CSV file")
        print("  C     - Capture landmarks for dataset (with label prompt)")
        print("  Q     - Quit")
        print("=" * 70)
        print("\n🎥 Starting camera...\n")
        
        landmarks_captured = 0
        prev_time = 0
        
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        ) as hands:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Failed to read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Calculate FPS
                current_time = cv2.getTickCount()
                fps = int(cv2.getTickFrequency() / (current_time - prev_time + 1))
                prev_time = current_time
                
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # Count hands
                num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
                
                # Draw hand landmarks with connections
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # Draw landmarks
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Label hand (Left/Right)
                        if results.multi_handedness:
                            hand_label = results.multi_handedness[idx].classification[0].label
                            # Get wrist position for label placement
                            wrist = hand_landmarks.landmark[0]
                            h, w, _ = frame.shape
                            cx, cy = int(wrist.x * w), int(wrist.y * h)
                            cv2.putText(frame, hand_label, (cx - 30, cy - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                
                # Draw overlay
                frame = self.draw_info_overlay(frame, num_hands, landmarks_captured, fps)
                
                # Display
                cv2.imshow("Hand Landmark Detector", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n✅ Exiting...")
                    break
                
                elif key == 32:  # SPACE
                    if results.multi_hand_landmarks:
                        landmarks = self.extract_landmarks(results)
                        print("\n" + "=" * 70)
                        print("📊 LANDMARKS (126 values):")
                        print("=" * 70)
                        print("Format: [Left_Hand_63_features, Right_Hand_63_features]")
                        print(" ".join(map(str, landmarks)))
                        print("=" * 70 + "\n")
                    else:
                        print("⚠️  No hands detected - try again")
                
                elif key == ord('s'):
                    if results.multi_hand_landmarks:
                        landmarks = self.extract_landmarks(results)
                        filename = self.save_landmarks_to_file(landmarks)
                        landmarks_captured += 1
                        print(f"💾 Landmarks saved to: {filename}")
                    else:
                        print("⚠️  No hands detected - cannot save")
                
                elif key == ord('c'):
                    if results.multi_hand_landmarks:
                        landmarks = self.extract_landmarks(results)
                        
                        # Prompt for label
                        print("\n" + "=" * 50)
                        label = input("Enter label (e.g., A, B, C...): ").strip().upper()
                        if label:
                            filename = self.save_landmarks_to_file(landmarks, label)
                            landmarks_captured += 1
                            print(f"✅ Labeled landmarks saved: {filename}")
                            print(f"   Label: {label}")
                            print("=" * 50 + "\n")
                        else:
                            print("⚠️  No label entered - skipping")
                    else:
                        print("⚠️  No hands detected - cannot capture")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Session Summary:")
        print(f"   Total landmarks captured: {landmarks_captured}")
        print(f"   Output directory: {self.output_dir}")
        print("\n✅ Program terminated successfully")

def main():
    detector = HandLandmarkDetector()
    detector.run()

if __name__ == "__main__":
    main()