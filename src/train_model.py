import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

DATA_PATH = "data/final_landmarks.csv"
MODEL_PATH = "models/rf_model.joblib"

df = pd.read_csv(DATA_PATH)

X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(str)

print("[INFO] Samples:", X.shape[0])
print("[INFO] Features:", X.shape[1])

assert X.shape[1] == 126, "Two-hand model requires 126 features"

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Accuracy: {acc*100:.2f}%")

os.makedirs("models", exist_ok=True)
joblib.dump(
    {"model": model, "label_encoder": le, "feature_size": 126},
    MODEL_PATH
)

print("✅ Two-hand model saved")