
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from model import MultiTaskModel  # 假设模型结构放在 model.py 中，或与训练脚本在同一目录

# 参数
MODEL_PATH = "model_baseline.pt"  # 修改为你保存的模型路径
DATA_PATH = "test.csv"  # 替换为测试集路径
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 128
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 定义 Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=MAX_LEN)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
        }

    def __len__(self):
        return len(self.encodings['input_ids'])

# 加载测试数据
df = pd.read_csv(DATA_PATH)
df = pd.read_csv(DATA_PATH).dropna(subset=['raw_text', 'sentiment', 'intensity_level'])
test_texts = df['raw_text'].tolist()
test_dataset = SentimentDataset(test_texts)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 加载模型
model = MultiTaskModel().to(DEVICE)
model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval()

# 推理
# ------------------ 推理部分 ------------------
predictions = []
pred_scores = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        logits, intensity = model(input_ids, attention_mask)

        # 累积全部结果
        pred_labels = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        pred_intensity = intensity.cpu().numpy().tolist()
        predictions.extend(pred_labels)
        pred_scores.extend(pred_intensity)

# ---------- 写入并评估 ----------
rounded_levels = [min(5, max(0, int(round(s)))) for s in pred_scores]

# 写入到 DataFrame
df['predicted_sentiment'] = predictions
df['predicted_intensity'] = pred_scores
df['predicted_intensity_level'] = rounded_levels

# ---------- 二分类：sentiment ----------
if 'sentiment' in df.columns:
    y_true_cls = df['sentiment'].tolist()
    y_pred_cls = predictions

    print("\n🎯 情感二分类指标：")
    print("Accuracy :", accuracy_score(y_true_cls, y_pred_cls))
    print("Precision:", precision_score(y_true_cls, y_pred_cls, average='binary'))
    print("Recall   :", recall_score(y_true_cls, y_pred_cls, average='binary'))
    print("F1-score :", f1_score(y_true_cls, y_pred_cls, average='binary'))
    print("详细报告：")
    print(classification_report(y_true_cls, y_pred_cls, target_names=["负面", "正面"]))

# ---------- 多等级分类：intensity_level ----------
if 'intensity_level' in df.columns:
    y_true_lvl = df['intensity_level'].tolist()
    y_pred_lvl = rounded_levels

    print("\n🎯 情感强度等级预测指标（四舍五入后）：")
    print("Accuracy :", accuracy_score(y_true_lvl, y_pred_lvl))
    print("Macro F1 :", f1_score(y_true_lvl, y_pred_lvl, average='macro'))
    print("Weighted F1:", f1_score(y_true_lvl, y_pred_lvl, average='weighted'))
    print("详细报告：")
    print(classification_report(y_true_lvl, y_pred_lvl, labels=[0,1,2,3,4,5]))


# ---------- 保存 ----------
df.to_csv("test_results.csv", index=False, encoding='utf-8-sig')
print("✅ 推理完成，结果已保存为 test_results.csv")
