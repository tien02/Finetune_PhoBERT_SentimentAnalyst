import config
import torch
import torch.nn as nn
from dataset import train_dataloader
from torch.optim import AdamW
from torchmetrics import Accuracy
from model import PhoBertClassifier
from pytorch_lightning import LightningModule

class PhoBERTTrainer(LightningModule):
    def __init__(self):
        super(PhoBERTTrainer, self).__init__()
        self.model = PhoBertClassifier()
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
    
    def training_step(self, batch):
        sent, input_ids, attn_mask = batch.values()

        logits = self.model(input_ids, attn_mask)

        loss = self.loss_fn(logits, sent)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        sent, input_ids, attn_mask = batch.values()
        logits = self.model(input_ids, attn_mask)

        loss = self.loss_fn(logits, sent)
        acc = self.accuracy(logits, sent)

        self.log("test_loss", loss)
        self.log("test_accuracy", acc)
        
        return loss
        

    def validation_step(self, batch, batch_idx):
        sent, input_ids, attn_mask = batch.values()
        logits = self.model(input_ids, attn_mask)

        loss = self.loss_fn(logits, sent)
        acc = self.accuracy(logits, sent)

        self.log("val_loss", loss)
        self.log("val_accuracy", acc)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=5e-5, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=config.EPOCHS)
        return {
            "optimizer":optimizer,
            "lr_scheduler": scheduler
        }