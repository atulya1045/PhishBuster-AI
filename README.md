# 🛡️ PhishBuster AI

**PhishBuster AI** is an intelligent phishing detection suite powered by machine learning.

🔗 **Live Demo:** [Click to Open Web App](https://phishbuster-ai-10.streamlit.app/)

---

## 🚀 Features

- 🔗 Detect **phishing URLs** using XGBoost
- 📧 Identify **email phishing** with BERT + Logistic Regression ensemble
- 📊 Visualize predictions using **SHAP explainability**
- 📁 Supports **bulk analysis** of URLs and emails via CSV
- 🎯 Built with **Streamlit** for a clean, interactive UI

---

## 📂 Project Structure

PhishBuster-AI/
├── app/ # Streamlit app logic
│ └── app.py
│
├── utils/ # Utility scripts (predict.py, shap_explain.py)
│
├── models/ # Trained ML models (XGBoost, Logistic Regression)
│
├── data/ # Sample/test datasets
│
├── notebooks/ # Jupyter notebooks for training & EDA
│
├── scripts/ # Scripts for model training, preprocessing
│
├── results/ # Output predictions, SHAP plots
│
├── config.py # Central configuration (if any)
├── requirements.txt # Python dependencies
├── .env # Environment variables (Git-ignored)
├── .gitignore # Files/folders to ignore in Git
└── README.md # Project documentation

🧠 Tech Stack
Python 3.9+

XGBoost, scikit-learn, BERT

SHAP for explainability

Streamlit for frontend

Pandas, Regex, Joblib

🔐 Cybersecurity Focus
PhishBuster AI contributes to cyber defense by:

Detecting deceptive phishing URLs

Identifying phishing emails based on content features

Offering transparency with SHAP to understand attack traits

📄 License
MIT License.
© 2025 Atulya Sawant