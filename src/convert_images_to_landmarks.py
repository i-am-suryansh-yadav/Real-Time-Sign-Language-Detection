import cv2
import mediapipe as mp
import os
import csv
import numpy as np

DATASETS = [
    "data/ISL_dataset_1",
    "data/ISL_dataset_2",
    "data/ISL_dataset_3"
]

OUTPUT_CSV = "data/combined_landmarks.csv"

mp_hands = mp.solutions.hands

def extract_two_hands(image):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        left = np.zeros(63, dtype=np.float32)
        right = np.zeros(63, dtype=np.float32)

        for i, hand in enumerate(results.multi_hand_landmarks[:2]):
            pts = []
            for lm in hand.landmark:
                pts.extend([lm.x, lm.y, lm.z])
            pts = np.array(pts, dtype=np.float32)

            if i == 0:
                left = pts
            else:
                right = pts

        return np.concatenate([left, right])

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"f{i}" for i in range(126)] + ["label"])

    for dataset in DATASETS:
        print(f"[INFO] Processing {dataset}")
        for label in os.listdir(dataset):
            class_dir = os.path.join(dataset, label)
            if not os.path.isdir(class_dir):
                continue

            for img in os.listdir(class_dir):
                path = os.path.join(class_dir, img)
                image = cv2.imread(path)
                if image is None:
                    continue

                features = extract_two_hands(image)
                if features is None:
                    continue

                writer.writerow(features.tolist() + [label])

print("✅ Two-hand landmark CSV created:", OUTPUT_CSV)