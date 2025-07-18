import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import jieba
import re

# 配置
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#清洗文本
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)           # 去除HTML标签
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text)  # 保留中英文和数字
    text = text.strip()
    return text

# 加载词典
def load_sentiment_dicts():
    def load_set(path):
        with open(path, encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())

    def load_dict(path):
        d = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    d[parts[0]] = float(parts[1])
        return d

    pos_words = load_set('dicts/pos_dict.txt')
    neg_words = load_set('dicts/neg_dict.txt')
    negation_words = load_set('dicts/denial_dict.txt')
    degree_dict = load_dict('dicts/adverb_dict.txt')
    return pos_words, neg_words, negation_words, degree_dict

# 提取情感分数
def extract_sentiment_score(text, pos_words, neg_words, negation_words, degree_dict):
    tokens = list(jieba.cut(text))
    i = 0
    score = 0
    while i < len(tokens):
        deg = 1.0
        flip = 1
        while i < len(tokens) and tokens[i] in degree_dict:
            deg *= degree_dict[tokens[i]]
            i += 1
        if i < len(tokens) and tokens[i] in negation_words:
            flip = -1
            i += 1
        if i < len(tokens):
            word = tokens[i]
            if word in pos_words:
                score += 1 * deg * flip
            elif word in neg_words:
                score -= 1 * deg * flip
        i += 1
    return score

# 加载数据
df = pd.read_csv("train.csv").dropna(subset=["raw_text", "sentiment", "intensity"])
pos_words, neg_words, negation_words, degree_dict = load_sentiment_dicts()
df["lexicon_score"] = df["raw_text"].apply(lambda x: extract_sentiment_score(x, pos_words, neg_words, negation_words, degree_dict))
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 数据集类
class SentimentDataset(Dataset):
    def __init__(self, dataframe):
        self.encodings = []
        self.labels = dataframe["sentiment"].tolist()
        self.scores = dataframe["intensity"].tolist()
        self.lexicon_feats = dataframe["lexicon_score"].tolist()

        for text in dataframe["raw_text"]:
            text = clean_text(text)
            tokens = jieba.lcut(text)
            encoding = tokenizer(" ".join(tokens), truncation=True, padding="max_length", max_length=MAX_LEN)
            self.encodings.append(encoding)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["score"] = torch.tensor(self.scores[idx], dtype=torch.float)
        item["lexicon_feat"] = torch.tensor(self.lexicon_feats[idx], dtype=torch.float).unsqueeze(0)
        return item

# 模型定义
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 1, 2)
        self.regressor = nn.Linear(self.bert.config.hidden_size + 1, 1)

    def forward(self, input_ids, attention_mask, lexicon_feat):
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        pooled = self.dropout(pooled)
        combined = torch.cat([pooled, lexicon_feat], dim=1)
        cls_out = self.classifier(combined)
        reg_out = self.regressor(combined).squeeze(-1)
        return cls_out, reg_out

# 训练主函数
def train_model():
    train_loader = DataLoader(SentimentDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SentimentDataset(val_df), batch_size=BATCH_SIZE)
    model = MultiTaskModel().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            score = batch["score"].to(DEVICE)
            feat = batch["lexicon_feat"].to(DEVICE)

            optimizer.zero_grad()
            cls_out, reg_out = model(ids, mask, feat)
            loss = cls_loss_fn(cls_out, label) + reg_loss_fn(reg_out, score)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                label = batch["label"].to(DEVICE)
                feat = batch["lexicon_feat"].to(DEVICE)
                cls_out, _ = model(ids, mask, feat)
                pred = torch.argmax(cls_out, dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        print(f"Validation Accuracy: {correct / total:.4f}")

    torch.save(model.state_dict(), "model_lexicon.pt")
    print("✅ 模型保存成功：model_lexicon.pt")

if __name__ == "__main__":
    train_model()
