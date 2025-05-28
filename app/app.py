import streamlit as st
import os
import sys
import json
import pandas as pd
from pathlib import Path
from utils.db_utils import log_prediction_to_db


# Set project root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Local imports
from utils.predict import (
    predict_url, predict_email, predict_bulk_urls, predict_bulk_emails
)
from utils.shap_explain import explain_email_prediction

# Paths
MODEL_DIR = project_root / "models"
META_PATH = MODEL_DIR / "xgboost_metadata.json"

# Load threshold
with open(META_PATH, "r") as f:
    metadata = json.load(f)
    URL_THRESHOLD = metadata.get("threshold", 0.5)

# Streamlit config
st.set_page_config(page_title="PhishBuster AI", layout="wide", page_icon="🛡️")
st.title("🛡️ PhishBuster AI - Intelligent Phishing Detection Suite")
st.markdown("""
Welcome to **PhishBuster AI** — a powerful phishing detection system powered by machine learning.
- Detect **phishing URLs** using XGBoost 🧠  
- Identify **email phishing attacks** with an ensemble of BERT & Logistic Regression 📧  
- Visualize model explanations via **SHAP** 🔍  
- Upload CSVs for **bulk analysis** 📁
""")

# Sidebar navigation
task = st.sidebar.radio("Select Task", [
    "🔗 URL Phishing Detection",
    "📧 Email Phishing Detection",
    "📁 Bulk Analysis",
    "📊 SHAP Explanation (Email)",
    "📈 Results Dashboard",
    "ℹ️ About"
])

# --- URL Detection --- #
if task == "🔗 URL Phishing Detection":
    st.header("🔗 Analyze a Suspicious URL")
    url_input = st.text_input("Enter a URL:")
    if st.button("Analyze URL"):
        if url_input:
            result = predict_url(url_input)
            st.success(f"✅ Prediction: {result['prediction']}")
            st.info(f"🔍 Confidence Score: {result['probability']}")
        else:
            st.warning("Please enter a valid URL.")


# --- Email Detection --- #
elif task == "📧 Email Phishing Detection":
    st.header("📧 Analyze an Email")
    email_input = st.text_area("Paste the email content below:", height=300)
    if st.button("Analyze Email"):
        if email_input:
            result = predict_email(email_input)
            st.success(f"✅ Prediction: {result['prediction']}")
            st.info(f"🔍 Confidence Score: {result['probability']}")
        else:
            st.warning("Please input email content.")

# --- Bulk Analysis --- #
elif task == "📁 Bulk Analysis":
    st.header("📁 Bulk Phishing Analysis")
    mode = st.selectbox("Choose mode", ["URLs", "Emails"])
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 File Preview:", df.head())

        if mode == "URLs":
            if "url" in df.columns:
                results = predict_bulk_urls(df["url"].tolist())
                st.write("🔍 Predictions:", results.head())
                st.download_button("📥 Download Results", results.to_csv(index=False), "url_predictions.csv")
            else:
                st.error("CSV must contain a 'url' column.")
        else:
            if "email" in df.columns:
                results = predict_bulk_emails(df["email"])
                st.write("🔍 Predictions:", results.head())
                st.download_button("📥 Download Results", results.to_csv(index=False), "email_predictions.csv")
            else:
                st.error("CSV must contain an 'email' column.")

# --- SHAP Explanation --- #
elif task == "📊 SHAP Explanation (Email)":
    st.header("📊 Explain Email Prediction with SHAP")
    sample_email = st.text_area("Paste email content to explain:", height=300)

    if st.button("Explain with SHAP"):
        if sample_email:
            explain_email_prediction(sample_email)
        else:
            st.warning("Please provide an email to explain.")

# --- Results Dashboard --- #
elif task == "📈 Results Dashboard":
    import dashboard
    st.header("📈 PhishBuster AI Results Dashboard")
    dashboard.run()

# --- About Section --- #
elif task == "ℹ️ About":
    st.header("ℹ️ About PhishBuster AI")
    st.markdown("""
**PhishBuster AI** is an ML-powered phishing detection system built with:
- XGBoost for phishing URL detection  
- A BERT + Logistic Regression ensemble for phishing emails  
- SHAP for transparent explainability  
- Feature engineering with adversarial and semantic features  

📫 Developed by Atulya Sawant  
🔒 PhishBuster AI © 2025 — Secure your digital life.
""")

st.markdown("---")
st.caption("🛡️ Powered by Streamlit & Machine Learning | All rights reserved.")
# End of app.py