import re
import numpy as np
import pandas as pd

def clean_url(url):
    return re.sub(r"http[s]?://", "", url).strip()

def predict_email(email_text, model):
    # Predict using the email ensemble model
    label_idx = model.predict([email_text])[0]
    prob = model.predict_proba([email_text])[0][label_idx]
    label = "Phishing" if label_idx == 1 else "Legitimate"
    return label, prob

def predict_url(url, model):
    # Feature engineering (you can expand this with more features)
    features = {
        "url_length": len(url),
        "has_https": int("https" in url),
        "num_dots": url.count("."),
        "has_at_symbol": int("@" in url),
        "has_hyphen": int("-" in url),
    }

    df = pd.DataFrame([features])
    label_idx = model.predict(df)[0]
    prob = model.predict_proba(df)[0][label_idx]
    label = "Phishing" if label_idx == 1 else "Legitimate"
    return label, prob
