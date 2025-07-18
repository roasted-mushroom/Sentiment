import torch
import torch.nn as nn
from transformers import BertModel
import jieba

# 模型结构（整合词典特征）
class MultiTaskModel(nn.Module):
    def __init__(self, model_name='bert-base-chinese', use_lexicon=True):
        super().__init__()
        self.use_lexicon = use_lexicon
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)

        input_dim = self.bert.config.hidden_size + (1 if use_lexicon else 0)
        self.classifier = nn.Linear(input_dim, 2)
        self.regressor = nn.Linear(input_dim, 1)

    def forward(self, input_ids, attention_mask, lexicon_feat=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)

        if self.use_lexicon and lexicon_feat is not None:
            combined = torch.cat([pooled_output, lexicon_feat], dim=1)
        else:
            combined = pooled_output

        classification_logits = self.classifier(combined)
        regression_output = self.regressor(combined).squeeze(-1)
        return classification_logits, regression_output

# 加载情感词典
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

# 提取文本情感词典得分
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
