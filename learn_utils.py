import torch
import torch.nn as nn
from transformers import BertModel
import jieba
import re

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
