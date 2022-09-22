import config
import torch.nn as nn
from transformers import AutoModel

class PhoBertClassifier(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(PhoBertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(config.CHECKPOINT)
        self.act = nn.Softmax(dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.1),
            nn.Linear(768, config.NUM_CLASSES))
        
        if freeze_backbone:
            for param in self.bert.parameters():
                param.require_grad = False
    
    def forward(self, input_ids, attn_mask):
        bert_feature = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        last_hidden_cls = bert_feature[0][:, 0, :]
        logits = self.classifier(last_hidden_cls)
        return self.act(logits)