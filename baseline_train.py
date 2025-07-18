import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 配置
MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
df = pd.read_csv("train.csv")  # 替换成你的文件路径
df = df[['raw_text', 'sentiment', 'intensity']].dropna()

# 使用原始文本作为训练输入
train_texts, val_texts, train_labels, val_labels, train_scores, val_scores = train_test_split(
    df['raw_text'].tolist(), df['sentiment'].tolist(), df['intensity'].tolist(), test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, scores):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=MAX_LEN)
        self.labels = labels
        self.scores = scores

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'score': torch.tensor(self.scores[idx], dtype=torch.float)
        }

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_texts, train_labels, train_scores)
val_dataset = SentimentDataset(val_texts, val_labels, val_scores)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 模型定义
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        classification_logits = self.classifier(pooled_output)
        regression_output = self.regressor(pooled_output)
        return classification_logits, regression_output.squeeze(-1)

model = MultiTaskModel().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn_class = nn.CrossEntropyLoss()
loss_fn_reg = nn.MSELoss()

# 训练函数
def train(model, loader):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        scores = batch['score'].to(DEVICE)

        optimizer.zero_grad()
        logits, preds = model(input_ids, attention_mask)
        loss_class = loss_fn_class(logits, labels)
        loss_reg = loss_fn_reg(preds, scores)
        loss = loss_class + loss_reg
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 验证函数
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            scores = batch['score'].to(DEVICE)

            logits, preds = model(input_ids, attention_mask)
            loss_class = loss_fn_class(logits, labels)
            loss_reg = loss_fn_reg(preds, scores)
            loss = loss_class + loss_reg
            total_loss += loss.item()

            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return total_loss / len(loader), accuracy

# 训练循环
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# 训练完成后保存模型
torch.save(model.state_dict(), "model_baseline.pt")
print("✅ 模型已保存为 model_baseline.pt")
