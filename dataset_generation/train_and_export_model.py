#exporting model of ogistic regression saving scalars and json

import os
import json
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------- Config ----------
DATA_PATH = "dataset/uciWomenBreastCancerData.csv"
OUT_DIR = "models"          # where we save scaler/model/json
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1) Load / clean ----------
print("[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"    ✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

print("[2/6] Cleaning dataset...")
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
print(f"    ✅ Cleaned dataset. Features: {df.shape[1]-1}, Target: 1 column")

# ---------- 2) Train/test split ----------
print("[3/6] Splitting into train/test sets...")
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    ✅ Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

# ---------- 3) Scale ----------
print("[4/6] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("    ✅ Features scaled")

# ---------- 4) Train logistic regression ----------
print("[5/6] Training Logistic Regression model...")
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train_scaled, y_train)
print("    ✅ Model trained")

# ---------- 5) Evaluate (sanity check) ----------
print("[6/6] Evaluating model...")
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"    ✅ Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------- 6) Save artifacts ----------
print("\n[EXPORT] Saving scaler, model (joblib), and JSON params...")

# 6a) Save full scaler & model (easy to reload for plaintext inference)
dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
dump(clf, os.path.join(OUT_DIR, "logreg.joblib"))

# 6b) Save minimal plaintext parameters (weights/bias + scaler params + feature order) -> useful for HE client
model_params = {
    "feature_names": X.columns.tolist(),                  # exact order of features
    "weights": clf.coef_.ravel().tolist(),                # flat list of floats
    "bias": float(clf.intercept_.ravel()[0]),            # scalar
    "scaler_mean": scaler.mean_.tolist(),                 # list
    "scaler_scale": scaler.scale_.tolist()                # list
}

with open(os.path.join(OUT_DIR, "model_params.json"), "w") as f:
    json.dump(model_params, f, indent=2)

print(f"    ✅ Saved joblib scaler -> {OUT_DIR}/scaler.joblib")
print(f"    ✅ Saved joblib model  -> {OUT_DIR}/logreg.joblib")
print(f"    ✅ Saved params json   -> {OUT_DIR}/model_params.json")

# ---------- 7) Verify JSON params reproduce same predictions (sanity check) ----------
print("\n[VERIFY] Loading saved JSON params to reproduce predictions (sanity check)...")
with open(os.path.join(OUT_DIR, "model_params.json"), "r") as f:
    mp = json.load(f)

weights = np.array(mp["weights"])
bias = float(mp["bias"])
mean = np.array(mp["scaler_mean"])
scale = np.array(mp["scaler_scale"])

# scale X_test using saved mean/scale
X_test_arr = X_test[mp["feature_names"]].astype(float).to_numpy()
X_test_scaled_from_json = (X_test_arr - mean) / scale

# compute logistic score, sigmoid and threshold
scores = X_test_scaled_from_json.dot(weights) + bias
probs = 1.0 / (1.0 + np.exp(-scores))
preds_from_json = (probs >= 0.5).astype(int)

acc_json = accuracy_score(y_test.to_numpy(), preds_from_json)
print(f"    ✅ Reproduced accuracy from JSON params: {acc_json:.4f}")

if np.array_equal(preds_from_json, y_pred):
    print("    ✅ Predictions from JSON params exactly match model.predict on X_test.")
else:
    same = (preds_from_json == y_pred).sum()
    print(f"    ⚠️ Predictions differ for {len(y_pred) - same} samples (floating-point rounding may cause tiny differences).")

print("\nAll done. Files are in the 'models' directory.")
