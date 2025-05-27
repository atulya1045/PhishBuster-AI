import pandas as pd
import joblib
import os
import re
import numpy as np
import pickle
from sklearn.metrics import classification_report

MODEL_PATH = 'models/logreg_tuned_spam_classifier.joblib'
ENCODER_PATH = 'models/label_encoder.joblib'
ADVERSARIAL_SAVE_PATH = 'models/adversarial_samples.pkl'

print("🚀 Starting adversarial test script...")

# === Load model and encoder
try:
    pipeline = joblib.load(MODEL_PATH)
    print("✅ Model loaded.")
    label_encoder = joblib.load(ENCODER_PATH)
    print("✅ Label encoder loaded.")
except Exception as e:
    print(f"❌ Error loading model or label encoder: {e}")
    exit()

# === Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("✅ Clean function ready.")

# === Adversarial test samples
adversarial_texts = [
    "Frëe cäsh! Click höre for a prize!",
    "Urgent!! Act now to win a gift card",
    "You have won a free ticket, claim it now!",
    "Hey, see you for brunch?",
    "Just sent the report. Let me know!",
    "Catch you later, don’t forget your keys!"
]

adversarial_labels = ['spam', 'spam', 'spam', 'ham', 'ham', 'ham']

print("✅ Adversarial texts defined.")

# === Clean and encode
cleaned_texts = [clean_text(t) for t in adversarial_texts]
true_labels = label_encoder.transform(adversarial_labels)
print("✅ Texts cleaned and labels encoded.")

# === Predict
predicted_labels = pipeline.predict(cleaned_texts)
print("✅ Predictions complete.")

# === Report
print("\n✅ Adversarial Classification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_))

# === Save results
results_df = pd.DataFrame({
    'original_text': adversarial_texts,
    'cleaned_text': cleaned_texts,
    'true_label': adversarial_labels,
    'predicted_label': label_encoder.inverse_transform(predicted_labels)
})

os.makedirs(os.path.dirname(ADVERSARIAL_SAVE_PATH), exist_ok=True)
with open(ADVERSARIAL_SAVE_PATH, 'wb') as f:
    pickle.dump(results_df, f)

print(f"\n✅ Adversarial samples saved to '{ADVERSARIAL_SAVE_PATH}'")
print("🚀 Adversarial test script completed successfully!")