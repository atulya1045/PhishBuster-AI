import os
import sys
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocessing import preprocess_data

# Paths to data and model
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_phishbuster.csv"))
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "random_forest_model.pkl"))

# Load and preprocess data
df = pd.read_csv(data_path)
df, label_col = preprocess_data(df)
X = df.drop(label_col, axis=1)
y = df[label_col]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load RandomForest model
rf_model = joblib.load(model_path)

# Predict probabilities
y_probs_rf = rf_model.predict_proba(X_test)[:, 1]

# =========================
# Use threshold = 0.55
# =========================
threshold = 0.55
y_pred_final = (y_probs_rf >= threshold).astype(int)

# Compute evaluation metrics
report = classification_report(y_test, y_pred_final, output_dict=True)
cm = confusion_matrix(y_test, y_pred_final)
auc = roc_auc_score(y_test, y_probs_rf)

print(f"\n=== RandomForest Final Evaluation (Threshold = {threshold}) ===")
print(f" Accuracy: {report['accuracy']:.4f}")
print(f" F1 (phishing): {report['1']['f1-score']:.4f}")
print(f" Recall (phishing): {report['1']['recall']:.4f}")
print(f" Precision (phishing): {report['1']['precision']:.4f}")
print(f" ROC AUC: {auc:.4f}")
print(" Confusion Matrix:")
print(cm)

# Optional: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"RandomForest (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - RandomForest')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
