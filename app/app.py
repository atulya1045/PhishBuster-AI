import streamlit as st
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

from utils.predict import predict_email, predict_url
from utils.shap_explain import explain_email_prediction

# Load models
EMAIL_MODEL_PATH = "models/logreg_bert_ensemble.joblib"
URL_MODEL_PATH = "models/xgboost_model.json"

email_model = joblib.load(EMAIL_MODEL_PATH)

# 🔄 Load XGBoost model from JSON (Streamlit-compatible)
url_model = xgb.XGBClassifier()
url_model.load_model(URL_MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="PhishBuster.AI", layout="wide")
st.title("🛡️ PhishBuster.AI – Intelligent Phishing Detector")

st.markdown("Detect phishing threats in emails or URLs using cutting-edge machine learning and explainable AI.")

mode = st.sidebar.selectbox("Choose Detection Mode", ["Phishing Email Detection", "Phishing URL Detection"])

if mode == "Phishing Email Detection":
    st.header("📧 Email Scanner")
    input_method = st.radio("Input Method", ["Paste Email Text", "Upload Email File (.txt)"])
    email_text = ""

    if input_method == "Paste Email Text":
        email_text = st.text_area("Paste your email content here", height=200)
    else:
        uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])
        if uploaded_file:
            email_text = uploaded_file.read().decode("utf-8")

    if email_text:
        st.subheader("🧠 Model Prediction")
        label, prob = predict_email(email_text, email_model)
        st.success(f"Predicted as: **{label}** with probability **{prob:.2f}**")

        # SHAP explanation
        st.subheader("🔍 Explainability (SHAP)")
        fig = explain_email_prediction(email_text, email_model)
        st.pyplot(fig)

elif mode == "Phishing URL Detection":
    st.header("🌐 URL Scanner")
    url_input = st.text_input("Enter a URL to scan")

    if url_input:
        st.subheader("🧠 Model Prediction")
        label, prob = predict_url(url_input, url_model)
        st.success(f"Predicted as: **{label}** with probability **{prob:.2f}**")

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit, BERT, SHAP & XGBoost")
st.sidebar.markdown("© 2023 PhishBuster.AI Team")
