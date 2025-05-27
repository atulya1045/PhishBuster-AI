import pandas as pd
import string
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove links
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation/numbers
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_spam_data(input_csv, output_csv):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"❌ File not found: {input_csv}")

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to read {input_csv}: {e}")

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("❌ CSV must have 'label' and 'text' columns.")

    df = df[["label", "text"]].dropna()

    # Normalize label values
    df["label"] = df["label"].str.lower().str.strip()

    # Debug check: unique labels
    valid_labels = ["ham", "spam"]
    found_labels = df["label"].unique().tolist()
    print(f"Found labels: {found_labels}")

    df = df[df["label"].isin(valid_labels)]

    if df.empty:
        raise ValueError("❌ No valid 'ham' or 'spam' labels found.")

    df["text_clean"] = df["text"].apply(clean_text)

    # Drop any rows with empty cleaned text
    df = df[df["text_clean"].str.strip().astype(bool)]

    if df.empty:
        raise ValueError("❌ No valid data after preprocessing.")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Preprocessing complete. Saved to: {output_csv}")

if __name__ == "__main__":
    input_path = "data/processed/cleaned_spam.csv"
    output_path = "data/processed/preprocessed_spam.csv"
    preprocess_spam_data(input_path, output_path)
