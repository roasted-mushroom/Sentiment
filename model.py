import torch.nn as nn
from transformers import BertModel

class MultiTaskModel(nn.Module):
    def __init__(self, model_name='bert-base-chinese'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        classification_logits = self.classifier(pooled_output)
        regression_output = self.regressor(pooled_output)
        return classification_logits, regression_output.squeeze(-1)
