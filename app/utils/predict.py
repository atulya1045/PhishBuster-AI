import numpy as np
import xgboost as xgb
from transformers import BertTokenizer, BertModel
import torch
import joblib

# Load tokenizer and BERT model once
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def extract_email_features(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding

def predict_email(email_text, email_model):
    features = extract_email_features(email_text)
    label = email_model.predict(features)[0]
    prob = email_model.predict_proba(features)[0][1]
    label_name = "Phishing" if label == 1 else "Legitimate"
    return label_name, prob

def extract_url_features(url):
    # You must match this to the features used in training
    features = {
        'url_length': len(url),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special_chars': sum(not c.isalnum() for c in url),
        'has_https': int("https" in url.lower()),
        'has_at_symbol': int("@" in url),
        'has_ip': int(any(c.isdigit() for c in url.split("/")[2] if url.startswith("http"))),
        'has_dash': int("-" in url),
        'has_double_slash': int("//" in url),
    }
    return np.array([list(features.values())])

def predict_url(url_text, url_model):
    features = extract_url_features(url_text)
    label = url_model.predict(features)[0]
    prob = url_model.predict_proba(features)[0][1]
    label_name = "Phishing" if label == 1 else "Legitimate"
    return label_name, prob
