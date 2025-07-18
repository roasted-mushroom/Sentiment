import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from lexicon_utils import MultiTaskModel, extract_sentiment_score, load_sentiment_dicts
import pandas as pd
from tqdm import tqdm
import os

MODEL_NAME = "bert-base-chinese"
BATCH_SIZE = 16
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, intensities, lexicon_feats, tokenizer):
        self.encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        self.labels = labels
        self.intensities = intensities
        self.lexicon_feats = lexicon_feats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["intensity"] = torch.tensor(self.intensities[idx], dtype=torch.float)
        item["lexicon_feat"] = torch.tensor([self.lexicon_feats[idx]], dtype=torch.float)
        return item

def train_model():
    df = pd.read_csv("train.csv")
    df = df.dropna(subset=["raw_text", "sentiment", "intensity_level"])
    texts = df["raw_text"].tolist()
    labels = df["sentiment"].tolist()
    intensities = df["intensity_level"].astype(float).tolist()

    # Load lexicons
    pos_words, neg_words, negation_words, degree_dict = load_sentiment_dicts()
    lexicon_feats = [extract_sentiment_score(text, pos_words, neg_words, negation_words, degree_dict) for text in texts]

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    dataset = SentimentDataset(texts, labels, intensities, lexicon_feats, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MultiTaskModel(model_name=MODEL_NAME, use_lexicon=True).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    ce_loss = nn.CrossEntropyLoss()
    reg_loss = nn.SmoothL1Loss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            intensity = batch["intensity"].to(DEVICE)
            lexicon_feat = batch["lexicon_feat"].to(DEVICE)

            optimizer.zero_grad()
            logits, intensity_pred = model(input_ids, attention_mask, lexicon_feat)
            loss_cls = ce_loss(logits, labels)
            loss_reg = reg_loss(intensity_pred, intensity)
            loss = loss_cls + 0.5 * loss_reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "model_lexicon_optimized.pt")
    print("✅ 模型已保存为 model_lexicon_optimized.pt")

if __name__ == "__main__":
    train_model()
