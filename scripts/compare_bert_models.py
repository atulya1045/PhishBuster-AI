import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

# === CONFIG ===
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "processed", "preprocessed_spam.csv")
bert_model_path = os.path.join(base_dir, "models", "bert_spam_model")
ensemble_model_path = os.path.join(base_dir, "models", "ensemble_bert_classifier.pt")

# === LOAD DATA ===
df = pd.read_csv(data_path)
X = df['text'].tolist()
y_true = df['label'].tolist()

# === Load BERT Model ===
tokenizer = BertTokenizer.from_pretrained(bert_model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(bert_model_path, local_files_only=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# === Tokenization & Prediction ===
def predict_bert(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1]  # Prob of class 1
    return probs.cpu().numpy()

print("\n🔍 Predicting with BERT...")
bert_probs = predict_bert(model, tokenizer, X)
bert_preds = (bert_probs >= 0.5).astype(int)

# === Load Ensemble Model (Simple Torch Model) ===
print("🔍 Predicting with Ensemble BERT...")
ensemble_model = torch.load(ensemble_model_path, map_location=device)
ensemble_model.eval()

# If ensemble needs features (e.g. from tokenizer), simulate same interface
ensemble_probs = predict_bert(ensemble_model, tokenizer, X)
ensemble_preds = (ensemble_probs >= 0.5).astype(int)

# === Evaluation Function ===
def evaluate_model(name, y_true, probs, preds):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, preds, digits=4))
    fpr, tpr, _ = roc_curve(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    return fpr, tpr, precision, recall, roc_auc, pr_auc

# === Evaluate ===
bert_fpr, bert_tpr, bert_prec, bert_rec, bert_auc, bert_pr_auc = evaluate_model("BERT", y_true, bert_probs, bert_preds)
ens_fpr, ens_tpr, ens_prec, ens_rec, ens_auc, ens_pr_auc = evaluate_model("Ensemble BERT", y_true, ensemble_probs, ensemble_preds)

# === Plot ROC Curve ===
plt.figure(figsize=(10, 5))
plt.plot(bert_fpr, bert_tpr, label=f"BERT (AUC = {bert_auc:.4f})")
plt.plot(ens_fpr, ens_tpr, label=f"Ensemble BERT (AUC = {ens_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "outputs", "roc_comparison.png"))
plt.close()

# === Plot Precision-Recall Curve ===
plt.figure(figsize=(10, 5))
plt.plot(bert_rec, bert_prec, label=f"BERT (PR AUC = {bert_pr_auc:.4f})")
plt.plot(ens_rec, ens_prec, label=f"Ensemble BERT (PR AUC = {ens_pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "outputs", "pr_comparison.png"))
plt.close()

print("\n✅ Comparison complete. ROC and PR curves saved in outputs/")
print("\n🔍 BERT Predictions:")