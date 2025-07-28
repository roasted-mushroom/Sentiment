import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from noise_utils import clean_text, extract_sentiment_score, load_sentiment_dicts
from noise_utils import MultiTaskModel

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 10
SAVE_DIR = "noise"
os.makedirs(SAVE_DIR, exist_ok=True)

# 损失函数列表
LOSS_FUNCTIONS = {
    "MSE": F.mse_loss,
    "L1Loss": F.l1_loss,
    "SmoothL1": F.smooth_l1_loss,
    "Huber": lambda input, target, reduction='none': F.huber_loss(input, target, delta=1.0, reduction=reduction)
}

NOISE_FILES = {
    "noise_5": "train_noisy_5.csv",
    "noise_10": "train_noisy_10.csv",
    "noise_20": "train_noisy_20.csv"
}

# 加载词典
pos_words, neg_words, negation_words, degree_dict = load_sentiment_dicts()

# 自定义数据集
class SentiDataset(Dataset):
    def __init__(self, df):
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.encodings, self.lexicons = [], []
        self.labels_cls = df["sentiment"].tolist()
        self.labels_reg = df["intensity_level"].tolist()
        self.is_noisy = df["noisy"].tolist() if "noisy" in df.columns else [0]*len(df)

        for text in df["raw_text"]:
            clean = clean_text(text)
            enc = tokenizer(clean, truncation=True, padding='max_length', max_length=MAX_LEN)
            self.encodings.append(enc)
            score = extract_sentiment_score(clean, pos_words, neg_words, negation_words, degree_dict)
            self.lexicons.append(score)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['lexicon_feat'] = torch.tensor(self.lexicons[idx], dtype=torch.float).unsqueeze(0)
        item['label_cls'] = torch.tensor(self.labels_cls[idx], dtype=torch.long)
        item['label_reg'] = torch.tensor(self.labels_reg[idx], dtype=torch.float)
        item['noisy'] = torch.tensor(self.is_noisy[idx], dtype=torch.bool)
        return item

    def __len__(self):
        return len(self.labels_cls)

def evaluate_model(model, dataloader):
    model.eval()
    preds_cls, trues_cls = [], []
    preds_reg, trues_reg = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            lexicon_feat = batch['lexicon_feat'].to(DEVICE)
            labels_cls = batch['label_cls'].to(DEVICE)
            labels_reg = batch['label_reg'].to(DEVICE)

            logits, scores = model(input_ids, attention_mask, lexicon_feat)
            preds_cls.extend(torch.argmax(logits, dim=1).cpu().tolist())
            trues_cls.extend(labels_cls.cpu().tolist())
            pred_values = scores.detach().cpu()
            if pred_values.ndim == 2 and pred_values.shape[1] == 1:
                pred_values = pred_values.squeeze(1)
            preds_reg.extend(pred_values.tolist())
            trues_reg.extend(labels_reg.cpu().tolist())

    acc = accuracy_score(trues_cls, preds_cls)
    f1 = f1_score(trues_cls, preds_cls, average='macro')
    mae = mean_absolute_error(trues_reg, preds_reg)
    mse = mean_squared_error(trues_reg, preds_reg)
    return acc, f1, mae, mse

def train_model(loss_name, noise_key):
    loss_func = LOSS_FUNCTIONS[loss_name]
    train_df = pd.read_csv(NOISE_FILES[noise_key]).dropna(subset=['raw_text', 'sentiment', 'intensity_level'])
    val_df = pd.read_csv("valid.csv").dropna(subset=['raw_text', 'sentiment', 'intensity_level'])

    train_dataset = SentiDataset(train_df)
    val_dataset = SentiDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = MultiTaskModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_mae = float('inf')
    best_model = None
    clean_losses, noisy_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_clean_loss, epoch_noisy_loss = [], []
        loop = tqdm(train_loader, desc=f"{loss_name}-{noise_key} Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            lexicon_feat = batch['lexicon_feat'].to(DEVICE)
            labels_cls = batch['label_cls'].to(DEVICE)
            labels_reg = batch['label_reg'].to(DEVICE)
            is_noisy = batch['noisy'].to(DEVICE)

            optimizer.zero_grad()
            logits, scores = model(input_ids, attention_mask, lexicon_feat)
            loss_cls = F.cross_entropy(logits, labels_cls)
            reg_loss_sample = loss_func(scores, labels_reg, reduction='none')
            conf_weight = torch.where(is_noisy, 0.7, 1.0).float()
            loss_reg = (reg_loss_sample * conf_weight).mean()
            loss = loss_cls + loss_reg
            loss.backward()
            optimizer.step()

            epoch_clean_loss.extend(reg_loss_sample[~is_noisy].detach().cpu().numpy())
            epoch_noisy_loss.extend(reg_loss_sample[is_noisy].detach().cpu().numpy())

        clean_losses.append(sum(epoch_clean_loss)/len(epoch_clean_loss) if epoch_clean_loss else 0)
        noisy_losses.append(sum(epoch_noisy_loss)/len(epoch_noisy_loss) if epoch_noisy_loss else 0)

        acc, f1, mae, mse = evaluate_model(model, val_loader)
        print(f"\n[Val] Acc: {acc:.3f}, F1: {f1:.3f}, MAE: {mae:.3f}, MSE: {mse:.3f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model.state_dict()

    # 保存最优模型
    model_path = os.path.join(SAVE_DIR, f"best_model_{loss_name}_{noise_key}.pt")
    torch.save(best_model, model_path)

    # 保存损失图
    plt.figure(figsize=(6,4))
    plt.plot(clean_losses, label='Clean')
    plt.plot(noisy_losses, label='Noisy')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title(f"Loss Curve: {loss_name} on {noise_key}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"loss_{loss_name}_{noise_key}.png"))
    plt.close()

if __name__ == '__main__':
    for loss_name in LOSS_FUNCTIONS:
        for noise_key in NOISE_FILES:
            train_model(loss_name, noise_key)
