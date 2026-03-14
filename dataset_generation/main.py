import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("[1/6] Loading dataset...")
df = pd.read_csv("dataset/uciWomenBreastCancerData.csv")
df.info()
print(f"    ✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

print("[2/6] Cleaning dataset...")
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
print(f"    ✅ Cleaned dataset. Features: {df.shape[1]-1}, Target: 1 column")

print("[3/6] Splitting into train/test sets...")
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    ✅ Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

print("[4/6] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("    ✅ Training data scaled")
X_test_scaled = scaler.transform(X_test)
print("    ✅ Test data scaled")

print("[5/6] Training Logistic Regression model...")
clf = LogisticRegression(random_state=42, max_iter=1000, verbose=1)
clf.fit(X_train_scaled, y_train)
print("    ✅ Model trained")

print("[6/6] Evaluating model...")
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"    ✅ Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))