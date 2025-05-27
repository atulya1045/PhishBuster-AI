# app/utils/predict.py

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import re

from .feature_engineering import extract_email_features  # Updated

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
URL_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
URL_META_PATH = os.path.join(MODEL_DIR, "xgboost_metadata.json")
EMAIL_MODEL_PATH = os.path.join(MODEL_DIR, "logreg_bert_ensemble.joblib")

# Load Threshold
with open(URL_META_PATH, "r") as f:
    METADATA = json.load(f)
    THRESHOLD = METADATA.get("threshold", 0.5)

# --- URL Feature Extraction ---
def extract_url_features(url):
    features = {
        "url_length": len(url),
        "num_dots": url.count('.'),
        "num_hyphens": url.count('-'),
        "has_https": int("https" in url),
        "has_at_symbol": int("@" in url),
        "has_ip": int(bool(re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", url))),
    }
    return pd.DataFrame([features])

# --- URL Prediction ---
def predict_url(url: str):
    model = xgb.XGBClassifier()
    model.load_model(URL_MODEL_PATH)

    features = extract_url_features(url)
    proba = model.predict_proba(features)[0][1]
    prediction = int(proba >= THRESHOLD)

    return {
        "url": url,
        "prediction": "Phishing" if prediction else "Legitimate",
        "probability": round(proba, 4),
        "features": features
    }

def predict_bulk_urls(urls):
    results = []
    for url in urls:
        try:
            results.append(predict_url(url))
        except Exception as e:
            results.append({"url": url, "error": str(e)})
    return pd.DataFrame(results)

# --- Email Prediction ---
def predict_email(text: str):
    model = joblib.load(EMAIL_MODEL_PATH)
    features = extract_email_features(text)
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    return {
        "prediction": "Phishing" if prediction else "Legitimate",
        "probability": round(proba, 4),
        "features": features
    }

def predict_bulk_emails(email_series):
    results = []
    for text in email_series:
        try:
            results.append(predict_email(text))
        except Exception as e:
            results.append({"email": text, "error": str(e)})
    return pd.DataFrame(results)
# --- Bulk Analysis ---