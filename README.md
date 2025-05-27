# 🛡️ PhishBuster AI

**PhishBuster AI** is an intelligent phishing detection suite powered by machine learning.

![PhishBuster Banner](banner.png) 

## 🚀 Features

- 🔗 Detect **phishing URLs** using XGBoost
- 📧 Identify **email phishing** with BERT + Logistic Regression ensemble
- 📊 Visualize predictions using **SHAP explainability**
- 📁 Supports **bulk analysis** of URLs and emails via CSV
- 🎯 Built with **Streamlit** for a clean, interactive UI

---

## 📦 Project Structure

PhishBuster-AI/
│
├── app/                # Streamlit app logic
│   └── app.py
│
├── utils/              # Utility scripts (predict.py, shap_explain.py)
│
├── models/             # Trained models (XGBoost, LogisticRegression)
│
├── data/               # Sample/test datasets
│
├── notebooks/          # Jupyter notebooks for training & EDA
│
├── scripts/            # Model training / preprocessing scripts
│
├── results/            # Output predictions & plots
│
├── config.py           # Central configuration
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (ignored by Git)
├── .gitignore          # Files to ignore in Git
└── README.md           # Project documentation


