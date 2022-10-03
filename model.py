import config
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class PhoBertFeedForward(nn.Module):
    def __init__(self, from_pretrained=True, freeze_backbone=False):
        super(PhoBertFeedForward, self).__init__()
        phobert_config = RobertaConfig.from_pretrained(config.CHECKPOINT)
        self.bert = RobertaModel(config=phobert_config)
        if from_pretrained:
          self.bert = RobertaModel.from_pretrained(config.CHECKPOINT)
        self.act = nn.LogSoftmax(dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(config.MID_HIDDEN_LAYER, config.MID_HIDDEN_LAYER),
            nn.Dropout(0.1),
            nn.Linear(config.MID_HIDDEN_LAYER, config.NUM_CLASSES))
        
        if freeze_backbone:
            for param in self.bert.parameters():
                param.require_grad = False
    
    def forward(self, input_ids, attn_mask):
        bert_feature = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        last_hidden_cls = bert_feature[0][:, 0, :]
        logits = self.classifier(last_hidden_cls)
        return self.act(logits)


class PhoBERTLSTM(nn.Module):
  def __init__(self, from_pretrained=True, freeze_backbone=False):
    super(PhoBERTLSTM, self).__init__()
    phobert_config = RobertaConfig.from_pretrained(config.CHECKPOINT)
    self.bert = RobertaModel(config=phobert_config)
    if from_pretrained:
        self.bert = RobertaModel.from_pretrained(config.CHECKPOINT)

    self.lstm = nn.LSTM(input_size=config.MID_HIDDEN_LAYER, 
                        hidden_size=config.MID_HIDDEN_LAYER, 
                        batch_first=True, bidirectional=True, num_layers=1)
    self.dropout = nn.Dropout(0.1)
    self.linear = nn.Linear(config.MID_HIDDEN_LAYER * 2, config.NUM_CLASSES)
    self.act = nn.LogSoftmax(dim=1)

    if freeze_backbone:
            for param in self.bert.parameters():
                param.require_grad = False
  
  def forward(self, input_ids, attn_mask):
    out = self.bert(input_ids=input_ids, attention_mask=attn_mask)[0]
    out, (h, c) = self.lstm(out)
    hidden = torch.cat((h[0], h[1]), dim = 1)
    out = self.linear(self.dropout(hidden))
    out = self.act(out)
    return out