import os
import argparse
import joblib

# === Set Paths
MODEL_PATH = os.path.join("models", "logreg_bert_ensemble.joblib")
ENCODER_PATH = os.path.join("models", "label_encoder.joblib")

# === Load Model & Encoder
print("📦 Loading model and label encoder...")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
print("✅ Model and label encoder loaded.")

# === Predict Function
def predict_label(text):
    pred = model.predict([text])[0]
    label = label_encoder.inverse_transform([pred])[0]
    return label

# === CLI Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Detection Inference")
    parser.add_argument("--text", type=str, required=True, help="Text/email to classify")
    args = parser.parse_args()

    prediction = predict_label(args.text)
    print(f"\n🧠 Prediction: {prediction.upper()}")
