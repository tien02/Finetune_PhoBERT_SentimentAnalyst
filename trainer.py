import config
import torch
import torch.nn as nn
from dataset import train_dataloader
from torch.optim import AdamW
from torchmetrics import Accuracy, Precision, Recall, F1Score
from model import PhoBertClassifier
from pytorch_lightning import LightningModule

class PhoBERTTrainer(LightningModule):
    def __init__(self):
        super(PhoBERTTrainer, self).__init__()
        self.model = PhoBertClassifier()
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(threshold=config.THRESHOLD)
        self.pre = Precision(threshold=config.THRESHOLD)
        self.re = Recall(threshold=config.THRESHOLD)
        self.f1 = F1Score(threshold=config.THRESHOLD)

    def forward(self, input_ids, attn_mask):
        return self.model(input_ids, attn_mask)
    
    def training_step(self, batch):
        sent, input_ids, attn_mask = batch.values()

        logits = self.model(input_ids, attn_mask)

        loss = self.loss_fn(logits, sent)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        sent, input_ids, attn_mask = batch.values()

        logits = self.model(input_ids, attn_mask)

        loss = self.loss_fn(logits, sent)

        acc = self.acc(logits, sent)
        pre = self.pre(logits, sent)
        recall = self.re(logits, sent)
        f1 = self.f1(logits, sent)
        
        self.log_dict({
            "loss": loss,
            "accuracy": acc,
            "precision": pre,
            "recall": recall,
            "f1_score": f1
        })
        

    def validation_step(self, batch, batch_idx):
        sent, input_ids, attn_mask = batch.values()

        logits = self.model(input_ids, attn_mask)

        loss = self.loss_fn(logits, sent)

        acc = self.acc(logits, sent)
        pre = self.pre(logits, sent)
        recall = self.re(logits, sent)
        f1 = self.f1(logits, sent)

        self.log_dict({
            "loss": loss,
            "accuracy": acc,
            "precision": pre,
            "recall": recall,
            "f1_score": f1
        })

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=5e-5, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                    steps_per_epoch=len(train_dataloader), epochs=config.EPOCHS)
        return {
            "optimizer":optimizer,
            "lr_scheduler": scheduler
        }