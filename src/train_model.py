import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter

DATA_PATH = os.path.join("data", "isl_landmarks.csv")
MODEL_PATH = os.path.join("models", "rf_model.joblib")

def load_data(path):
    df = pd.read_csv(path)
    # Try to detect label column
    if 'label' in df.columns:
        X = df.drop(columns=['label']).values
        y = df['label'].values
    else:
        # assume first column is label otherwise
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
    return X, y, df

def simple_oversample_minimum(X, y, min_count=2, random_state=42):
    """
    If some classes have < min_count samples, duplicate random samples
    from that class until it has min_count. Returns new X,y.
    """
    np.random.seed(random_state)
    counter = Counter(y)
    rows = []
    for cls, cnt in counter.items():
        if cnt < min_count:
            # indices belonging to class
            idxs = np.where(y == cls)[0]
            # choose samples with replacement until min_count
            needed = min_count - cnt
            chosen = np.random.choice(idxs, size=needed, replace=True)
            for c in chosen:
                rows.append((X[c].copy(), cls))
    if rows:
        X_extra = np.vstack([r[0] for r in rows])
        y_extra = np.array([r[1] for r in rows])
        X_new = np.vstack([X, X_extra])
        y_new = np.concatenate([y, y_extra])
        return X_new, y_new
    else:
        return X, y

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Put isl_landmarks.csv in data/")

    X, y, df = load_data(DATA_PATH)
    print(f"Loaded data: X shape = {X.shape}, y shape = {y.shape}")
    counts = Counter(y)
    print("Label distribution (label:count):")
    for k,v in sorted(counts.items()):
        print(f"  {k}: {v}")

    # If any class has less than 2 samples, we have two options:
    # 1) Do non-stratified split (may imbalance test/train)
    # 2) Oversample the classes with <2 samples to reach a minimum (so stratify works)
    min_count = min(counts.values())
    use_stratify = True
    if min_count < 2:
        print("\nWARNING: Some classes have fewer than 2 samples; stratified split would fail.")
        # We will attempt a very small oversample to reach 2 samples per class
        X_os, y_os = simple_oversample_minimum(X, y, min_count=2)
        if X_os.shape[0] != X.shape[0]:
            print(f"  Oversampled: new X shape = {X_os.shape}, new label counts:")
            from collections import Counter as C2
            for k,v in sorted(C2(y_os).items()):
                print(f"    {k}: {v}")
            X, y = X_os, y_os
            use_stratify = True
        else:
            # fallback: do not stratify
            print("  Could not oversample (unexpected) — will proceed WITHOUT stratify.")
            use_stratify = False

    # encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # split
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, shuffle=True
        )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # train model
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Test accuracy: {acc*100:.2f}%")

    # Only run cross-val if every class has >= 2 samples
    counts_after = Counter(le.inverse_transform(np.unique(y_enc)))
    if min_count >= 2:
        try:
            cv = cross_val_score(model, X, y_enc, cv=5)
            print(f"5-fold CV accuracies: {cv}")
            print(f"5-fold CV mean accuracy: {cv.mean()*100:.2f}%")
        except Exception as e:
            print("Could not run cross_val_score:", e)
    else:
        print("Skipping cross-validation because some classes still have < 2 samples.")

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()