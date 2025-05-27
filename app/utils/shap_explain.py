# app/utils/shap_explain.py

import shap
import pandas as pd
import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt

from .feature_engineering import extract_email_features  # Updated import

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "logreg_bert_ensemble.joblib")

def explain_email_prediction(email_text: str):
    st.subheader("🧠 SHAP Explanation for Email Prediction")
    st.markdown("Understand **why** the model predicted this email as phishing or legitimate.")

    # Load model
    model = joblib.load(MODEL_PATH)

    # Extract features
    features = extract_email_features(email_text)
    
    # Explain
    explainer = shap.Explainer(model.predict_proba, features)
    shap_values = explainer(features)

    st.markdown("### 🔍 Feature Contribution (Waterfall Plot)")
    fig1 = shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig1)

    st.markdown("### 📊 Feature Importance (Bar Chart)")
    fig2 = shap.plots.bar(shap_values, show=False)
    st.pyplot(fig2)
