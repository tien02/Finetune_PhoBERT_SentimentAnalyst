import config
import torch.nn as nn
from termcolor import colored
from pyvi import ViTokenizer
from transformers import AutoModel, AutoTokenizer

class PhoBertClassifier(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(PhoBertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(config.CHECKPOINT)
        self.classifier = nn.Sequential(
            nn.Linear(768, config.MID_HIDDEN_LAYER),
            nn.ReLU(),
            nn.Linear(config.MID_HIDDEN_LAYER, config.NUM_CLASSES))
        
        if freeze_backbone:
            for param in self.bert.parameters():
                param.require_grad = False
    
    def forward(self, input_ids, attn_mask):
        bert_feature = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        last_hidden_cls = bert_feature[0][:, 0, :]
        logits = self.classifier(last_hidden_cls)
        return logits

# def test(num_classes=3):
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
#     sentence = "Xin chào, tôi là Tiến"
#     tokens = tokenizer(ViTokenizer.tokenize(sentence), return_tensors='pt', padding=True)
#     phobert = PhoBertClassifier(num_classes)
#     phobert.eval()
#     logits = phobert(tokens['input_ids'], tokens['attn_mask'])

#     assert logits.size(1) == num_classes, "Ouput classes did not match number of classes"
#     print(colored("Success!", "green"))

# if __name__ == '__main__':
    # test()