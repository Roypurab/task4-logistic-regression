"""
Task 4: Classification with Logistic Regression
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
    - loaded directly from sklearn.datasets (same dataset as the UCI/Kaggle
      "Breast Cancer Wisconsin" CSV that was linked in the task).
    - A CSV copy is also saved locally as breast_cancer_data.csv so the raw
      data file is present in the repo, as required by the submission rules.

Steps performed:
    1. Load a binary classification dataset (malignant=1 / benign=0... here
       we keep sklearn's native encoding: 0 = malignant, 1 = benign)
    2. Train/test split + feature standardization
    3. Fit a Logistic Regression model
    4. Evaluate with confusion matrix, precision, recall, ROC-AUC
    5. Tune the decision threshold and visualize/explain the sigmoid function
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, roc_auc_score, precision_recall_curve, classification_report
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame  # features + target column named 'target'
df.to_csv("breast_cancer_data.csv", index=False)

print("Dataset shape:", df.shape)
print("Target classes:", dict(zip(data.target_names, [0, 1])))
print("Class balance:\n", df["target"].value_counts(), "\n")

X = df.drop(columns=["target"])
y = df["target"]  # 0 = malignant, 1 = benign

# ---------------------------------------------------------------------------
# 2. TRAIN/TEST SPLIT + STANDARDIZATION
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. FIT LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)

# Predicted probabilities for the positive class (1 = benign)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred_default = model.predict(X_test_scaled)  # uses default 0.5 threshold

# ---------------------------------------------------------------------------
# 4. EVALUATION AT DEFAULT THRESHOLD (0.5)
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_default)
acc = accuracy_score(y_test, y_pred_default)
prec = precision_score(y_test, y_pred_default)
rec = recall_score(y_test, y_pred_default)
f1 = f1_score(y_test, y_pred_default)
auc = roc_auc_score(y_test, y_proba)

print("=" * 60)
print("EVALUATION AT DEFAULT THRESHOLD = 0.5")
print("=" * 60)
print("Confusion Matrix:\n", cm)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nFull classification report:\n", classification_report(y_test, y_pred_default, target_names=data.target_names))

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix (threshold = 0.5)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# ROC CURVE
# ---------------------------------------------------------------------------
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. SIGMOID FUNCTION VISUALIZATION
# ---------------------------------------------------------------------------
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(6, 4))
plt.plot(z, sigmoid, color="teal")
plt.axhline(0.5, color="grey", linestyle="--", linewidth=1)
plt.axvline(0, color="grey", linestyle="--", linewidth=1)
plt.xlabel("z (linear combination: w·x + b)")
plt.ylabel("sigmoid(z) = P(y=1)")
plt.title("Sigmoid Function")
plt.tight_layout()
plt.savefig("sigmoid_function.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# THRESHOLD TUNING
# ---------------------------------------------------------------------------
precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(7, 5))
plt.plot(pr_thresholds, precisions[:-1], label="Precision", color="blue")
plt.plot(pr_thresholds, recalls[:-1], label="Recall", color="red")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("threshold_tuning.png", dpi=150)
plt.close()

# Try a few candidate thresholds and report metrics for each
print("\n" + "=" * 60)
print("THRESHOLD TUNING EXPERIMENTS")
print("=" * 60)
candidate_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []
for t in candidate_thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    p = precision_score(y_test, y_pred_t)
    r = recall_score(y_test, y_pred_t)
    f = f1_score(y_test, y_pred_t)
    a = accuracy_score(y_test, y_pred_t)
    results.append((t, a, p, r, f))
    print(f"Threshold={t:.1f} | Accuracy={a:.3f} | Precision={p:.3f} | Recall={r:.3f} | F1={f:.3f}")

threshold_df = pd.DataFrame(results, columns=["threshold", "accuracy", "precision", "recall", "f1"])
threshold_df.to_csv("threshold_tuning_results.csv", index=False)

print("\nAll plots and result files saved in the current directory.")
print("Files: confusion_matrix.png, roc_curve.png, sigmoid_function.png,")
print("       threshold_tuning.png, threshold_tuning_results.csv, breast_cancer_data.csv")
