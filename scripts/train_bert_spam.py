import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


data_path = "data/processed/cleaned_spam.csv"

# Verify cleaned data exists and is valid
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ ERROR: File not found: {data_path}")

# Load cleaned data
df = pd.read_csv(data_path)
df.dropna(subset=["cleaned_text", "label"], inplace=True)
df = df[df["cleaned_text"].str.strip() != ""]

# Check data integrity
if df.empty:
    raise ValueError("❌ ERROR: No valid data found after filtering. Check your cleaned_spam.csv file.")

# Encode labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

print("✅ Using device:", "cuda" if torch.cuda.is_available() else "cpu")
print("📊 Label distribution:")
print(df["label"].value_counts())

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["cleaned_text"].tolist(), df["label_encoded"].tolist(), test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset wrapper
class SpamDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SpamDataset(train_texts, train_labels)
val_dataset = SpamDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model setup
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(set(train_labels)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = Adam(model.parameters(), lr=5e-5)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)

# Training function
def train(model, train_loader, val_loader, epochs=3):
    for epoch in range(epochs):
        print(f"\n📘 Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc="🔁 Training")
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        print(f"✅ Epoch {epoch + 1} training loss: {total_loss / len(train_loader):.4f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        accuracy = correct / total
        print(f"📈 Validation Accuracy: {accuracy:.4f}")

# Start training
train(model, train_loader, val_loader)