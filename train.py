import torch
import torch.nn as nn
import config
import dataset
from trainer import set_seed, trainer, init_model

train_loader = dataset.train_dataloader
val_loader = dataset.eval_dataloader
test_loader = dataset.test_dataloader

set_seed(69)
loss_fn = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_classifier, optimizer, scheduler = init_model(device, train_loader, config.WARM_UP_EPOCH)
trainer(model=bert_classifier, train_dataloader=train_loader, val_dataloader=val_loader,
        device=device, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
        epoch=config.EPOCH, log_interval =config.LOG_INTERVAL, evaluation=config.EVALUATE_WHILE_TRAINING)