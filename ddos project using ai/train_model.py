"""
Train a Random Forest classifier for DDoS attack detection
and save the model + preprocessing artifacts.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ── 1. Load data ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset_sdn.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

print("Loading dataset …")
data = pd.read_csv(DATA_PATH)
print(f"  Shape: {data.shape}")

# ── 2. Pre-process ───────────────────────────────────────────
# Drop IP address columns (high cardinality strings, not useful for a simple RF)
data = data.drop(columns=["src", "dst"])

# Encode the Protocol column (categorical → numeric)
le_protocol = LabelEncoder()
data["Protocol"] = le_protocol.fit_transform(data["Protocol"].astype(str))

# Fill missing values (rx_kbps and tot_kbps have 506 nulls each)
data = data.fillna(data.median(numeric_only=True))

# Separate features and target
X = data.drop(columns=["label"])
y = data["label"]

feature_names = list(X.columns)
print(f"  Features ({len(feature_names)}): {feature_names}")
print(f"  Protocol classes: {list(le_protocol.classes_)}")

# ── 3. Train / test split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

# ── 4. Train model ───────────────────────────────────────────
print("Training Random Forest …")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── 5. Evaluate ──────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n  Accuracy: {acc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

# ── 6. Save artifacts ────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "ddos_model.joblib")
encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
features_path = os.path.join(MODEL_DIR, "feature_names.joblib")

joblib.dump(model, model_path)
joblib.dump(le_protocol, encoder_path)
joblib.dump(feature_names, features_path)

print(f"\n  Model saved   → {model_path}")
print(f"  Encoder saved → {encoder_path}")
print(f"  Features saved→ {features_path}")
print("\nDone ✓")
