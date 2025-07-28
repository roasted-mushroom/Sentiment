import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from transformers import BertTokenizer
from learn_utils import clean_text, extract_sentiment_score, load_sentiment_dicts, MultiTaskModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 256
BATCH_SIZE = 16

# 评估目标
LOSS_TYPES = ["MSE", "L1Loss", "SmoothL1", "Huber"]
NOISE_TAGS = ["noise_5", "noise_10", "noise_20"]

TEST_CSV = "test.csv"
SAVE_DIR = "noise"
os.makedirs(SAVE_DIR, exist_ok=True)

# 加载词典
pos_words, neg_words, negation_words, degree_dict = load_sentiment_dicts()

class SentiTestDataset(Dataset):
    def __init__(self, texts):
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.encodings, self.lexicons = [], []
        for text in texts:
            clean = clean_text(text)
            enc = tokenizer(clean, truncation=True, padding='max_length', max_length=MAX_LEN)
            self.encodings.append(enc)
            score = extract_sentiment_score(clean, pos_words, neg_words, negation_words, degree_dict)
            self.lexicons.append(score)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['lexicon_feat'] = torch.tensor(self.lexicons[idx], dtype=torch.float).unsqueeze(0)
        return item

    def __len__(self):
        return len(self.encodings)

def evaluate_model(loss_type, noise_tag):
    model_path = f"model_{loss_type}_{noise_tag}.pt"
    if not os.path.exists(model_path):
        print(f"❌ Missing model: {model_path}, skipped.")
        return

    print(f"\n🔍 Evaluating: {model_path}")
    df = pd.read_csv(TEST_CSV).dropna(subset=['raw_text', 'sentiment', 'intensity_level'])
    texts = df['raw_text'].tolist()
    test_dataset = SentiTestDataset(texts)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MultiTaskModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    pred_sentiment, pred_intensity = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            lexicon_feat = batch['lexicon_feat'].to(DEVICE)
            logits, scores = model(input_ids, attention_mask, lexicon_feat)
            pred_sentiment.extend(torch.argmax(logits, dim=1).cpu().tolist())
            pred_intensity.extend(scores.cpu().tolist())

    df['pred_sentiment'] = pred_sentiment
    df['pred_intensity'] = pred_intensity
    df['pred_intensity_level'] = [min(5, max(0, round(x))) for x in pred_intensity]
    df['fit_error'] = df['pred_intensity'] - df['intensity_level']

    # 指标
    y_true_cls = df['sentiment']
    y_pred_cls = df['pred_sentiment']
    acc_cls = accuracy_score(y_true_cls, y_pred_cls)
    f1_cls = f1_score(y_true_cls, y_pred_cls, average="binary")

    y_true_lvl = df['intensity_level']
    y_pred_lvl = df['pred_intensity_level']
    acc_lvl = accuracy_score(y_true_lvl, y_pred_lvl)
    f1_lvl = f1_score(y_true_lvl, y_pred_lvl, average="macro")

    y_reg = df['pred_intensity']
    mae = mean_absolute_error(y_true_lvl, y_reg)
    mse = mean_squared_error(y_true_lvl, y_reg)
    r2 = r2_score(y_true_lvl, y_reg)

    # 打印并保存图
    print(f"Sentiment ACC: {acc_cls:.3f} | F1: {f1_cls:.3f}")
    print(f"Intensity ACC: {acc_lvl:.3f} | F1: {f1_lvl:.3f}")
    print(f"MAE: {mae:.3f} | MSE: {mse:.3f} | R2: {r2:.3f}")

    plt.figure(figsize=(5,5))
    plt.scatter(y_true_lvl, y_reg, alpha=0.4, label='Predictions')
    plt.plot([0,5], [0,5], 'r--', label='Perfect Fit')
    plt.xlabel("True Intensity Level")
    plt.ylabel("Predicted Intensity")
    plt.title(f"Fit - {loss_type} / {noise_tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"fit_{loss_type}_{noise_tag}.png"))
    plt.close()

    plt.figure(figsize=(5,4))
    plt.hist(df['fit_error'], bins=30, color='gray', edgecolor='black')
    plt.title("Residual Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"residual_{loss_type}_{noise_tag}.png"))
    plt.close()

    df.to_csv(os.path.join(SAVE_DIR, f"result_{loss_type}_{noise_tag}.csv"), index=False)

# 主程序入口
if __name__ == '__main__':
    for loss_type in LOSS_TYPES:
        for noise_tag in NOISE_TAGS:
            evaluate_model(loss_type, noise_tag)
