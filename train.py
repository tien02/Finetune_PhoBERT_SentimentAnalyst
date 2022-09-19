import torch
import torch.nn as nn
import config
import dataset
from termcolor import colored
from trainer import set_seed, trainer, init_model

train_loader = dataset.train_dataloader
val_loader = dataset.eval_dataloader
test_loader = dataset.test_dataloader

set_seed(69)
loss_fn = nn.CrossEntropyLoss()
print(colored(f"Use device [{config.DEVICE}]", 'green'))
bert_classifier, optimizer, scheduler = init_model(config.DEVICE, train_loader, config.WARM_UP_EPOCH)
trainer(model=bert_classifier, train_dataloader=train_loader, val_dataloader=val_loader,
        device=config.DEVICE, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
        epochs=config.EPOCH, log_interval=config.LOG_INTERVAL, evaluation=config.EVALUATE_WHILE_TRAINING)