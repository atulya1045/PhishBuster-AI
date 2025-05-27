import os
import sys
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Import the class instead of function
from utils.preprocessing import URLPreprocessor

# Paths
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_phishbuster.csv"))
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Load and preprocess data
df = pd.read_csv(data_path)
preprocessor = URLPreprocessor()
df, label_col = preprocessor.preprocess_data(df)
X = df.drop(label_col, axis=1)
y = df[label_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load models
models = {
    'RandomForest': joblib.load(os.path.join(models_dir, 'random_forest_model.pkl')),
    'XGBoost': joblib.load(os.path.join(models_dir, 'xgboost_model.pkl')),
    'LogisticRegression': joblib.load(os.path.join(models_dir, 'logistic_model.pkl'))
}

# Fixed threshold
fixed_threshold = 0.5

# Define evaluation with fixed threshold
def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, proba)
    return report, auc, cm, proba

# Evaluate each model
metrics = []
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        report, auc, cm, y_prob = evaluate_with_threshold(model, X_test, y_test, threshold=fixed_threshold)
        metrics.append((name, fixed_threshold, report, auc, cm, y_prob))
    else:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc = None
        y_prob = None
        metrics.append((name, fixed_threshold, report, auc, cm, y_prob))

# Print summary
print("\n=== Model Comparison with Fixed Threshold (0.5) ===")
for name, threshold, report, auc, cm, _ in metrics:
    print(f"\nModel: {name} (Threshold: {threshold})")
    print(f" Accuracy: {report['accuracy']:.4f}")
    print(f" F1 (phishing): {report['1']['f1-score']:.4f}")
    print(f" Recall (phishing): {report['1']['recall']:.4f}")
    print(f" Precision (phishing): {report['1']['precision']:.4f}")
    if auc is not None:
        print(f" ROC AUC: {auc:.4f}")
    print(" Confusion Matrix:")
    print(cm)

# Plot ROC curves
plt.figure(figsize=(8, 6))
for name, _, _, auc, _, y_prob in metrics:
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Save metrics to CSV