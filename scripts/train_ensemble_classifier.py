import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import re

# === Load Dataset
DATA_PATH = 'data/processed/cleaned_spam.csv'
df = pd.read_csv(DATA_PATH)

# === Encode Labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("✅ Label mapping used:", label_mapping)

# === Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_encoded'], test_size=0.2, random_state=42, stratify=df['label_encoded']
)

# === TF-IDF Vectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, stop_words='english')

# === Base Models
logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
nb = MultinomialNB()
svc = LinearSVC(class_weight='balanced', max_iter=1000)

# === Ensemble Model
ensemble = VotingClassifier(
    estimators=[
        ('lr', logreg),
        ('nb', nb),
        ('svc', svc)
    ],
    voting='hard'
)

# === Pipeline
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('ensemble', ensemble)
])

# === Train
pipeline.fit(X_train, y_train)

# === Evaluate
y_pred = pipeline.predict(X_test)
print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === Save model and label encoder
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, 'models/logreg_bert_ensemble.joblib')
joblib.dump(label_encoder, 'models/label_encoder.joblib')
print("\n✅ Ensemble model and label encoder saved to 'models/'")

# === Adversarial Evaluation
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

adversarial_texts = [
    "Frëe cäsh! Click höre for a prize!",
    "Urgent!! Act now to win a gift card",
    "You have won a free ticket, claim it now!",
    "Hey, see you for brunch?",
    "Just sent the report. Let me know!",
    "Catch you later, don’t forget your keys!"
]
adversarial_labels = ['spam', 'spam', 'spam', 'ham', 'ham', 'ham']
cleaned_texts = [clean_text(t) for t in adversarial_texts]
true_labels = label_encoder.transform(adversarial_labels)
predicted_labels = pipeline.predict(cleaned_texts)

# === Report
print("\n✅ Adversarial Classification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_))

# === Save Adversarial Results
adversarial_results = pd.DataFrame({
    'original_text': adversarial_texts,
    'cleaned_text': cleaned_texts,
    'true_label': adversarial_labels,
    'predicted_label': label_encoder.inverse_transform(predicted_labels)
})
with open('models/adversarial_samples.pkl', 'wb') as f:
    pickle.dump(adversarial_results, f)

print("\n✅ Adversarial samples saved to 'models/adversarial_samples.pkl'")
# === End of script