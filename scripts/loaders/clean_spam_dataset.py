import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    if pd.isnull(text):
        return "none"
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def load_raw_file(filepath):
    """
    Attempt to load CSV using different encodings until one works.
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, engine='python', on_bad_lines='skip')
            print(f"✅ Loaded using encoding: {enc}")
            return df
        except Exception as e:
            print(f"❌ Failed with encoding {enc}: {e}")
    raise ValueError("❌ Could not read the file with known encodings.")

def detect_label_and_text_columns(df):
    label_col = None
    text_col = None

    for col in df.columns:
        col_data = df[col].astype(str).str.lower()

        if col_data.isin(['ham', 'spam']).sum() > 10:
            label_col = col
        elif df[col].apply(lambda x: isinstance(x, str) and len(x) > 20).sum() > 10:
            if not text_col:
                text_col = col

    if not label_col or not text_col:
        print("⚠️ Columns detected:\n", df.columns)
        raise ValueError("❌ Could not detect appropriate 'label' and 'text' columns.")

    return label_col, text_col

def load_and_clean_spam_dataset(filepath):
    df = load_raw_file(filepath)
    print(f"✅ Raw shape: {df.shape}")

    label_col, text_col = detect_label_and_text_columns(df)
    df = df[[label_col, text_col]]
    df.columns = ['label', 'text']
    print(f"✅ Detected columns -> label: {label_col}, text: {text_col}")

    df = df[df['label'].astype(str).str.lower().isin(['ham', 'spam'])]
    print(f"✅ Filtered rows: {df.shape[0]}")
    df = df.dropna(subset=['text'])
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print("✅ Cleaning done. Sample:")
    print(df[['label', 'cleaned_text']].head())
    return df

if __name__ == "__main__":
    input_path = "data/raw/spam.csv"
    output_path = "data/processed/cleaned_spam.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_cleaned = load_and_clean_spam_dataset(input_path)
    df_cleaned.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to: {output_path}")
