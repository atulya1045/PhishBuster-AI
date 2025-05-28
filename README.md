
---

```markdown
# 🛡️ **PhishBuster AI**

**PhishBuster AI** is an intelligent phishing detection suite powered by machine learning.

[![PhishBuster Web App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)](https://phishbuster-ai-10.streamlit.app/)

---

## 🚀 **Features**

- 🔗 Detect **phishing URLs** using XGBoost  
- 📧 Identify **email phishing** with BERT + Logistic Regression ensemble  
- 📊 Visualize predictions using **SHAP explainability**  
- 📁 Supports **bulk analysis** of URLs and emails via CSV  
- 🎯 Built with **Streamlit** for a clean, interactive UI  

---

## 📁 **Project Structure**

```

PhishBuster-AI/
├── app/                # Streamlit app logic
│   └── app.py
│
├── utils/              # Utility scripts (predict.py, shap\_explain.py)
│
├── models/             # Trained ML models (XGBoost, Logistic Regression)
│
├── data/               # Sample/test datasets
│
├── notebooks/          # Jupyter notebooks for training & EDA
│
├── scripts/            # Scripts for model training, preprocessing
│
├── results/            # Output predictions, SHAP plots
│
├── config.py           # Central configuration
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (Git-ignored)
├── .gitignore          # Files to ignore in Git
└── README.md           # Project documentation

````

---

## 🧠 **Tech Stack**

- **Python 3.9+**  
- [XGBoost](https://xgboost.ai), [scikit-learn](https://scikit-learn.org), [BERT](https://huggingface.co)  
- [SHAP](https://shap.readthedocs.io/en/latest/) for explainability  
- [Streamlit](https://streamlit.io/) for the frontend  
- Pandas, Regex, Joblib, JSON, OS, and other standard libraries  

---

## 🔐 **Cybersecurity Focus**

PhishBuster AI contributes to cyber defense by:

- 🧪 Detecting **phishing URLs** based on lexical & semantic features  
- 📩 Identifying **email phishing attempts** based on textual patterns and language models  
- 🧬 Providing **model transparency** using SHAP to understand threat characteristics  

---

## 🖥️ **How to Run Locally**

```bash
# Clone the repository
git clone https://github.com/atulya1045/PhishBuster-AI.git
cd PhishBuster-AI

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py
````

---

## ☁️ **Deployment**

This app is deployed on **Streamlit Cloud**:
🔗 [https://phishbuster-ai-10.streamlit.app](https://phishbuster-ai-10.streamlit.app)

---

## 🧪 **Sample Inputs**

* Sample **URLs** and **emails** for testing are located in the `data/` directory.
* Use the **Bulk Analysis** tab in the app to upload your own CSV.

---

## 📌 **Future Enhancements**

* 🧠 Add transformer fine-tuning for the email model
* 🌐 Real-time URL threat feeds integration
* 🧩 Browser extension for live phishing alerts
* 🛠️ Logging system for monitoring and alerting
* ✉️ Email header analysis (DKIM/SPF/DMARC) integration

---

## 📄 **License**

MIT License © 2025 Atulya Sawant

```

