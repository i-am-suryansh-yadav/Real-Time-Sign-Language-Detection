import pandas as pd

csv = pd.read_csv("data/combined_landmarks.csv")
csv.to_csv("data/final_landmarks.csv", index=False)

print("✅ final_landmarks.csv ready")