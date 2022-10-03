import config
import torch
import torch.nn as nn
from dataset import train_dataloader
from torch.optim import Adam
from torchmetrics import Accuracy, F1Score
from pytorch_lightning import LightningModule

class PhoBERTModel(LightningModule):
    def __init__(self, model):
        super(PhoBERTModel, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = Accuracy(threshold=config.THRESHOLD)
        self.f1 = F1Score(threshold=config.THRESHOLD)

    def forward(self, input_ids, attn_mask):
        return self.model(input_ids, attn_mask)
    
    def training_step(self, batch):
        sent, input_ids, attn_mask = batch.values()

        logits = self.model(input_ids, attn_mask)

        loss = self.loss_fn(logits, sent)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        sent, input_ids, attn_mask = batch.values()

        logits = self.model(input_ids, attn_mask)

        loss = self.loss_fn(logits, sent)

        acc = self.acc(logits, sent)
        f1 = self.f1(logits, sent)
        
        self.log_dict({
            "test_loss": loss,
            "test_accuracy": acc,
            "test_f1": f1,
        }, on_epoch=True, prog_bar=True)
        

    def validation_step(self, batch, batch_idx):
        sent, input_ids, attn_mask = batch.values()

        logits = self.model(input_ids, attn_mask)

        loss = self.loss_fn(logits, sent)

        acc = self.acc(logits, sent)
        f1 = self.f1(logits, sent)

        self.log_dict({
            "val_loss": loss,
            "val_accuracy": acc,
            "val_f1": f1,
        }, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=1e-4, eps=1e-6, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,
                    steps_per_epoch=len(train_dataloader), epochs=config.EPOCHS)
        return {
            "optimizer":optimizer,
            "lr_scheduler": scheduler
        }