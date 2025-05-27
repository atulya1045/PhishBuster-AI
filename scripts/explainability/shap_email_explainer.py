import os
import joblib
import shap
import numpy as np
import warnings
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning)

print("📦 Loading model and label encoder...")

# Load model and label encoder
model_path = "models/logreg_tuned_spam_classifier.joblib"
label_encoder_path = "models/label_encoder.joblib"

if not os.path.exists(model_path):
    sys.exit(f"❌ Model not found at: {model_path}")
if not os.path.exists(label_encoder_path):
    sys.exit(f"❌ Label encoder not found at: {label_encoder_path}")

logreg_model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

print("✅ Model and label encoder loaded.")

# Emails to explain
emails_to_explain = [
    "Free cäsh! Clïck here to claim your prize!",
    "Hey, I sent over the updated files — let me know.",
    "Urgent: Account access locked. Click to unlock!"
]

# Background texts
background_texts = [
    "Hello, I hope you are doing well.",
    "Please see the attached report and review.",
    "You have won a lottery! Click to claim now.",
    "We need your credentials to verify your account."
]

# ✅ Use SHAP Text masker
try:
    print("⚙️  Creating SHAP explainer with Text masker...")
    masker = shap.maskers.Text()
    explainer = shap.Explainer(logreg_model.predict_proba, masker)
except Exception as e:
    sys.exit(f"❌ Failed to initialize explainer: {e}")

# ✅ Explain emails
print("⚙️  Computing SHAP values...")
shap_values = explainer(emails_to_explain)

# ✅ Display results
try:
    from IPython.display import display
    for i, email in enumerate(emails_to_explain):
        print(f"\n🔍 Explanation for email {i + 1}: {email}")
        display(shap.plots.text(shap_values[i]))
except ImportError:
    print("⚠️ Install IPython for visual plots: pip install ipython")
    for i, email in enumerate(emails_to_explain):
        print(f"\n📝 Email {i + 1}: {email}")
        print("SHAP values:", shap_values[i].values)
    