# src/extract_landmarks.py
# Modified from uploaded script: Now supports 3 datasets (ISL_dataset1,2,3), two-hand detection (up to 126 features with padding),
# handles 'nothing' folder (forces zero features), and suppresses MediaPipe warnings.
# Processes A-Z + nothing folders, skips invalid images.

import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import logging

# Suppress MediaPipe and TensorFlow warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors

# ================= MEDIAPIPE SETUP =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,  # Changed to 2 for two-hand support
    min_detection_confidence=0.5  # Slightly lowered for better detection
)

# ================= DATASET PATHS =================
DATASET_PATHS = [
    "data/ISL_dataset1",
    "data/ISL_dataset2",
    "data/ISL_dataset3"
]

# ================= STORAGE =================
data = []
labels = []

# ================= PROCESS DATASETS =================
for dataset_path in DATASET_PATHS:
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path not found: {dataset_path}")
        continue
    
    print(f"\nProcessing dataset: {dataset_path}")

    for letter in sorted(os.listdir(dataset_path)):
        letter_folder = os.path.join(dataset_path, letter)

        if not os.path.isdir(letter_folder):
            continue

        print(f"  Letter: {letter}")

        if letter.upper() == 'NOTHING':
            # Special handling for 'nothing': Force zero features for each image file
            image_count = 0
            for img_name in os.listdir(letter_folder):
                img_path = os.path.join(letter_folder, img_name)
                if os.path.isfile(img_path):  # Only process files
                    # Append zero features (126 for two hands)
                    zero_features = np.zeros(126)
                    data.append(zero_features)
                    labels.append('nothing')
                    image_count += 1
            print(f"    Added {image_count} zero-feature samples for 'nothing'")
        else:
            # Normal processing for A-Z: Extract landmarks, skip if no hand
            image_count = 0
            for img_name in os.listdir(letter_folder):
                img_path = os.path.join(letter_folder, img_name)

                image = cv2.imread(img_path)
                if image is None:
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    # Extract features for up to 2 hands
                    landmark_list = []
                    for hand_landmarks in results.multi_hand_landmarks[:2]:
                        hand_pts = []
                        for lm in hand_landmarks.landmark:
                            hand_pts.extend([lm.x, lm.y, lm.z])
                        landmark_list.extend(hand_pts)
                    
                    # Pad to exactly 126 features if less than 2 hands
                    while len(landmark_list) < 126:
                        landmark_list.append(0.0)
                    
                    # Truncate if more (unlikely)
                    landmark_list = landmark_list[:126]
                    
                    if len(landmark_list) == 126:
                        data.append(landmark_list)
                        labels.append(letter)
                        image_count += 1
            
            print(f"    Extracted {image_count} valid images for {letter}")

# ================= SAVE DATA =================
if not data:
    print("No data extracted! Check dataset paths and images.")
else:
    df = pd.DataFrame(data)
    df["label"] = labels

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/landmarks.csv", index=False)

    print("\n✅ Landmark extraction completed!")
    print(f"Total samples extracted: {len(df)}")
    print("Saved to: data/landmarks.csv")
    print(f"Labels distribution:\n{df['label'].value_counts().sort_index()}")