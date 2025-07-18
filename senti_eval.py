import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import jieba
import re

from lexicon_utils import MultiTaskModel, extract_sentiment_score, load_sentiment_dicts

# 参数
MODEL_PATH = "model_lexicon.pt"
DATA_PATH = "test.csv"
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 128
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
pos_words, neg_words, negation_words, degree_dict = load_sentiment_dicts()

#清洗文本
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)           # 去除HTML标签
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text)  # 保留中英文和数字
    text = text.strip()
    return text

# 定义 Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts):
        self.encodings = []
        self.lexicon_feats = []
        for text in texts:
            text = clean_text(text)
            tokens = jieba.lcut(text)
            encoding = tokenizer(" ".join(tokens), truncation=True, padding='max_length', max_length=MAX_LEN)
            self.encodings.append(encoding)
            score = extract_sentiment_score(text, pos_words, neg_words, negation_words, degree_dict)
            self.lexicon_feats.append(score)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item["lexicon_feat"] = torch.tensor(self.lexicon_feats[idx]).unsqueeze(0)
        return item

    def __len__(self):
        return len(self.encodings)

# 加载测试数据
df = pd.read_csv(DATA_PATH)
df = pd.read_csv(DATA_PATH).dropna(subset=['raw_text', 'sentiment', 'intensity_level'])
texts = df['raw_text'].tolist()
test_dataset = SentimentDataset(texts)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 加载模型
model = MultiTaskModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 推理
predictions, pred_scores = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        lexicon_feat = batch['lexicon_feat'].to(DEVICE)
        logits, intensity = model(input_ids, attention_mask, lexicon_feat)

        pred_labels = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        pred_intensity = intensity.cpu().numpy().tolist()
        predictions.extend(pred_labels)
        pred_scores.extend(pred_intensity)

# 写入到 DataFrame
rounded_levels = [min(5, max(0, int(round(s)))) for s in pred_scores]
df['predicted_sentiment'] = predictions
df['predicted_intensity'] = pred_scores
df['predicted_intensity_level'] = rounded_levels

# 情感二分类评估
if 'sentiment' in df.columns:
    y_true_cls = df['sentiment'].tolist()
    print("\n🎯 情感二分类指标：")
    print("Accuracy :", accuracy_score(y_true_cls, predictions))
    print("Precision:", precision_score(y_true_cls, predictions, average='binary'))
    print("Recall   :", recall_score(y_true_cls, predictions, average='binary'))
    print("F1-score :", f1_score(y_true_cls, predictions, average='binary'))
    print("详细报告：")
    print(classification_report(y_true_cls, predictions, target_names=["负面", "正面"]))

# 情感强度等级评估
if 'intensity_level' in df.columns:
    y_true_lvl = df['intensity_level'].tolist()
    print("\n🎯 情感强度等级预测指标（四舍五入后）：")
    print("Accuracy :", accuracy_score(y_true_lvl, rounded_levels))
    print("Macro F1 :", f1_score(y_true_lvl, rounded_levels, average='macro'))
    print("Weighted F1:", f1_score(y_true_lvl, rounded_levels, average='weighted'))
    print("详细报告：")
    print(classification_report(y_true_lvl, rounded_levels, labels=[0,1,2,3,4,5]))

df.to_csv("test_results_lexicon.csv", index=False, encoding='utf-8-sig')
print("✅ 推理完成，结果已保存为 test_results_lexicon.csv")
