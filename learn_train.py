import os
import re
import jieba
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from torch.amp import GradScaler

# 配置
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 文本清洗函数，只保留中文
def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    return text

# 加载情感词典
def load_sentiment_dicts():
    with open("dicts/pos_dict.txt", encoding="utf-8") as f:
        pos_words = set(f.read().splitlines())
    with open("dicts/neg_dict.txt", encoding="utf-8") as f:
        neg_words = set(f.read().splitlines())
    with open("dicts/denial_dict.txt", encoding="utf-8") as f:
        negation_words = set(f.read().splitlines())
    degree_dict = {}
    with open("dicts/adverb_dict.txt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                word, weight = line.strip().split()
                degree_dict[word] = float(weight)
    return pos_words, neg_words, negation_words, degree_dict

# 提取文本的情感得分
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

# 自定义Dataset
class SentiDataset(Dataset):
    def __init__(self, encodings, labels_cls, labels_reg, lexicons):
        self.encodings = encodings
        self.labels_cls = labels_cls
        self.labels_reg = labels_reg
        self.lexicons = lexicons

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels_cls'] = torch.tensor(self.labels_cls[idx])
        item['labels_reg'] = torch.tensor(self.labels_reg[idx], dtype=torch.float)
        item['lexicon_feat'] = torch.tensor(self.lexicons[idx], dtype=torch.float).unsqueeze(0)
        return item

    def __len__(self):
        return len(self.labels_cls)

# 多任务模型：分类 + 回归
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 1, 2)
        self.regressor = nn.Linear(self.bert.config.hidden_size + 1, 1)

    def forward(self, input_ids, attention_mask, lexicon_feat):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        combined = torch.cat([pooled, lexicon_feat], dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        intensity = self.regressor(combined)
        return logits, intensity.squeeze(-1)

# Focal Loss（用于情感分类）
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

# Class-aware 回归损失
class ClassAwareMSELoss(nn.Module):
    def forward(self, preds, targets):
        return F.mse_loss(preds, targets)

# 主训练函数
def train_model():
    df = pd.read_csv("train.csv")
    texts = [clean_text(t) for t in df["raw_text"].tolist()]
    labels_cls = df["sentiment"].tolist()
    labels_reg = df["intensity_level"].tolist()

    # 加载情感词典
    pos_words, neg_words, negation_words, degree_dict = load_sentiment_dicts()
    lexicons = [extract_sentiment_score(t, pos_words, neg_words, negation_words, degree_dict) for t in texts]

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN)

    dataset = SentiDataset(encodings, labels_cls, labels_reg, lexicons)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = MultiTaskModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn_cls = FocalLoss()
    loss_fn_reg = ClassAwareMSELoss()
    scaler = GradScaler()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_cls = batch["labels_cls"].to(DEVICE)
            labels_reg = batch["labels_reg"].to(DEVICE)
            lexicon_feat = batch["lexicon_feat"].to(DEVICE)

            optimizer.zero_grad()
            logits, intensity = model(input_ids, attention_mask, lexicon_feat)
            loss_cls = loss_fn_cls(logits, labels_cls)
            loss_reg = loss_fn_reg(intensity, labels_reg)
            loss = loss_cls + loss_reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "model_learn.pt")
    print("✅ 模型保存成功：model_learn.pt")

if __name__ == "__main__":
    train_model()
