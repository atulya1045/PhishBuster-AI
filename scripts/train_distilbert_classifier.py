import os
import pandas as pd
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

# ✅ Load dataset
file_path = os.path.join("data", "processed", "cleaned_spam.csv")
print(f"✅ Using dataset: {file_path}")

df = pd.read_csv(file_path)
df = df.loc[:, ~df.columns.duplicated()]

# ✅ Fix column names
if 'cleaned_text' in df.columns and 'text' not in df.columns:
    df.rename(columns={'cleaned_text': 'text'}, inplace=True)

if not all(col in df.columns for col in ['label', 'text']):
    raise ValueError("CSV must contain 'label' and 'text' columns.")

# ✅ Convert string labels to integers
# Example: {'ham': 0, 'spam': 1}
if df['label'].dtype == 'object' or df['label'].dtype.name == 'category':
    label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
    df['label'] = df['label'].map(label_map)
    print(f"✅ Label mapping used: {label_map}")

# ✅ Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ✅ Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# ✅ Convert to Dataset and tokenize
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize)
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ✅ Load model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# ✅ TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# ✅ Train
trainer.train()
# ✅ Save model