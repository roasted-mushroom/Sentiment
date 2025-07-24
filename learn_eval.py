import os
import re
import torch
import jieba
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from learn_utils import clean_text, extract_sentiment_score, load_sentiment_dicts, MultiTaskModel

MODEL_PATH = "model_learn.pt"
DATA_PATH = "test.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 256
BATCH_SIZE = 16

# 加载词典
pos_words, neg_words, negation_words, degree_dict = load_sentiment_dicts()

# 数据集
class SentiTestDataset(Dataset):
    def __init__(self, texts):
        self.encodings = []
        self.lexicons = []
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        for text in texts:
            clean = clean_text(text)
            encoding = tokenizer(clean, truncation=True, padding='max_length', max_length=MAX_LEN)
            self.encodings.append(encoding)
            score = extract_sentiment_score(clean, pos_words, neg_words, negation_words, degree_dict)
            self.lexicons.append(score)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['lexicon_feat'] = torch.tensor(self.lexicons[idx], dtype=torch.float).unsqueeze(0)
        return item

    def __len__(self):
        return len(self.encodings)

# 加载数据
df = pd.read_csv(DATA_PATH).dropna(subset=['raw_text', 'sentiment', 'intensity_level'])
texts = df['raw_text'].tolist()
test_dataset = SentiTestDataset(texts)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 加载模型
model = MultiTaskModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 推理
pred_sentiment, pred_intensity = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        lexicon_feat = batch["lexicon_feat"].to(DEVICE)

        logits, scores = model(input_ids, attention_mask, lexicon_feat)
        pred_sentiment.extend(torch.argmax(logits, dim=1).cpu().tolist())
        pred_intensity.extend(scores.cpu().tolist())

# 整理输出
df["predicted_sentiment"] = pred_sentiment
df["predicted_intensity"] = pred_intensity
df["predicted_intensity_level"] = [min(5, max(0, round(s))) for s in pred_intensity]

# 情感分类评估
y_true_cls = df["sentiment"].tolist()
print("🎯 情感分类指标：")
print("Accuracy :", accuracy_score(y_true_cls, pred_sentiment))
print("F1-score :", f1_score(y_true_cls, pred_sentiment, average="binary"))
print(classification_report(y_true_cls, pred_sentiment, target_names=["负面", "正面"]))

# 情感强度等级评估
y_true_lvl = df["intensity_level"].tolist()
print("\n🎯 情感强度等级预测指标（四舍五入后）：")
print("Accuracy :", accuracy_score(y_true_lvl, df["predicted_intensity_level"].tolist()))
print("F1-score :", f1_score(y_true_lvl, df["predicted_intensity_level"].tolist(), average="macro"))
print(classification_report(y_true_lvl, df["predicted_intensity_level"].tolist(), labels=[0,1,2,3,4,5]))

# 保存结果
df.to_csv("test_results_learn.csv", index=False, encoding="utf-8-sig")
print("✅ 推理完成，结果保存为 test_results_learn.csv")
