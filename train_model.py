import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target

print("Dataset: Iris")
print(f"  Features : {iris.feature_names}")
print(f"  Classes  : {list(iris.target_names)}")
print(f"  Samples  : {X.shape[0]}")
print(f"  Features : {X.shape[1]}")
print()

# --------------------------------------------------
# 2. Split into Train / Test
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train samples : {len(X_train)}")
print(f"Test  samples : {len(X_test)}")
print()

# --------------------------------------------------
# 3. Preprocess (scale features)
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --------------------------------------------------
# 4. Train Model
# --------------------------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model trained: RandomForestClassifier (n_estimators=100)")

# --------------------------------------------------
# 5. Evaluate
# --------------------------------------------------
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# --------------------------------------------------
# 6. Save Model Artifacts
# --------------------------------------------------
# Bundle model + scaler + metadata into one file
model_artifact = {
    "model": model,
    "scaler": scaler,
    "feature_names": iris.feature_names,
    "target_names": list(iris.target_names),
    "accuracy": acc,
}

joblib.dump(model_artifact, "model.joblib")
print("Model saved → model.joblib")

# Quick sanity-check: reload and predict one sample
loaded = joblib.load("model.joblib")
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # known setosa
sample_scaled = loaded["scaler"].transform(sample)
pred = loaded["model"].predict(sample_scaled)[0]
print(f"\nSanity check — input {sample[0].tolist()} → predicted class: "
      f"'{loaded['target_names'][pred]}'  (expected: 'setosa')")
