import streamlit as st
import joblib
import matplotlib.pyplot as plt

from utils.predict import predict_email, predict_url
from utils.shap_explain import explain_email_prediction

# ───────────────────────────────────────────────
# Load Models
# ───────────────────────────────────────────────
EMAIL_MODEL_PATH = "models/logreg_bert_ensemble.joblib"
URL_MODEL_PATH = "models/xgboost_model.pkl"

email_model = joblib.load(EMAIL_MODEL_PATH)
url_model = joblib.load(URL_MODEL_PATH)

# ───────────────────────────────────────────────
# Page Config
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="PhishBuster.AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────
# Custom Styles
# ───────────────────────────────────────────────
st.markdown("""
    <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: bold;
            color: #2E8BFF;
        }
        .sub-title {
            font-size: 1.2rem;
            color: #666;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
        }
        .stRadio > div {
            gap: 10px;
        }
        .prediction-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin-top: 20px;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Title & Intro
# ───────────────────────────────────────────────
st.markdown("<div class='main-title'>🛡️ PhishBuster.AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-Powered Phishing Detection with Explainable Insights</div>", unsafe_allow_html=True)
st.markdown("Detect phishing emails and malicious URLs using machine learning and SHAP explainability.")

# Sidebar Branding
with st.sidebar:
    st.header("🧭 Navigation")
    page = st.radio("Choose Mode", ["📧 Email Detector", "🌐 URL Scanner"])
    st.markdown("---")
    st.info("💡 Tip: Use realistic phishing content for best results.")
    st.markdown("🚀 Version: 1.2.0")
    st.caption("© 2025 PhishBuster.AI")

# ───────────────────────────────────────────────
# EMAIL DETECTION
# ───────────────────────────────────────────────
if page == "📧 Email Detector":
    st.subheader("📧 Intelligent Email Phishing Detector")

    col1, col2 = st.columns([1, 2])
    with col1:
        input_method = st.radio("Select Input Method", ["Paste Email", "Upload .txt File"])
    with col2:
        st.markdown("")

    email_text = ""
    if input_method == "Paste Email":
        email_text = st.text_area("✉️ Enter email content", height=200)
    else:
        uploaded_file = st.file_uploader("📂 Upload email text file", type=["txt"])
        if uploaded_file:
            try:
                email_text = uploaded_file.read().decode("utf-8")
            except:
                st.error("🚫 Error reading the file.")

    if email_text:
        st.markdown("### 🔍 Detection Results")
        with st.spinner("Analyzing email with BERT & Logistic Regression..."):
            label, prob = predict_email(email_text, email_model)

        alert_color = "green" if label.lower() == "not phishing" else "red"
        st.markdown(
            f"<div class='prediction-box'>"
            f"<h5>📢 Prediction: <span style='color:{alert_color};'>{label.upper()}</span></h5>"
            f"<p>Model Confidence:</p>"
            f"<div style='background:#ddd;border-radius:10px;width:100%;'>"
            f"<div style='width:{prob*100:.0f}%;background:#2E8BFF;height:18px;border-radius:10px;'></div>"
            f"</div><p style='font-size:0.9rem;color:#888;'>{prob:.2f} probability</p></div>",
            unsafe_allow_html=True
        )

        st.markdown("### 🧠 SHAP Explanation")
        fig = explain_email_prediction(email_text, email_model)
        st.pyplot(fig)

# ───────────────────────────────────────────────
# URL DETECTION
# ───────────────────────────────────────────────
elif page == "🌐 URL Scanner":
    st.subheader("🌐 Real-time URL Threat Analyzer")

    url_input = st.text_input("🔗 Enter a suspicious URL")

    if url_input:
        st.markdown("### 🔍 Detection Results")
        with st.spinner("Scanning URL with XGBoost model..."):
            label, prob = predict_url(url_input, url_model)

        alert_color = "green" if label.lower() == "not phishing" else "red"
        st.markdown(
            f"<div class='prediction-box'>"
            f"<h5>📢 Prediction: <span style='color:{alert_color};'>{label.upper()}</span></h5>"
            f"<p>Model Confidence:</p>"
            f"<div style='background:#ddd;border-radius:10px;width:100%;'>"
            f"<div style='width:{prob*100:.0f}%;background:#2E8BFF;height:18px;border-radius:10px;'></div>"
            f"</div><p style='font-size:0.9rem;color:#888;'>{prob:.2f} probability</p></div>",
            unsafe_allow_html=True
        )

# ───────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────
st.markdown("---")
st.caption("🔬 This is a research-grade project — not a commercial tool.")
st.markdown("📚 For educational purposes only. Use responsibly.")