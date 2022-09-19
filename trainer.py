import config
import random 
import numpy as np
import torch

from torch.optim import AdamW
from model import PhoBertClassifier
from transformers import get_linear_schedule_with_warmup
from termcolor import colored

def set_seed(seed_value=69):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

def init_model(device, train_dataloader, epochs=4):
    bert_classifier = PhoBertClassifier(freeze_backbone=False)
    bert_classifier.to(device)
    optimizer = AdamW(bert_classifier.parameters(), lr=5e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=0, num_training_steps=total_steps)

    return bert_classifier, optimizer, scheduler

def trainer(model, train_dataloader, val_dataloader, device, loss_fn, optimizer, scheduler, epochs, log_interval, evaluation=False):
    '''
    Train PhoBertClassifier
    '''
    print(colored("Start Training...", "blue"))
    for epoch in range(epochs):
        print(colored(f"Epoch {epoch}", "yellow"))
        total_loss, batch_loss, batch_count = 0, 0, 0
        model.train()

        for idx, batch in enumerate(train_dataloader):
            batch_count += 1
            sent, input_ids, attn_mask = batch.values()

            attn_mask = attn_mask.to(device)
            input_ids = input_ids.to(device)
            sent = sent.to(device)
            
            model.zero_grad()
            logits = model(input_ids, attn_mask)

            loss = loss_fn(logits, sent)
            batch_loss += loss.detach()
            total_loss += loss.detach()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if idx % log_interval == 0:
                print(f"Batch [{idx}|{len(train_dataloader)}]: Loss {batch_loss / batch_count:.2f}")
                batch_loss, batch_count = 0, 0
        
        avg_loss = total_loss / len(train_dataloader)
        print(colored(f"Total loss is: {avg_loss}", "green"))
        if evaluation:
            val_loss, val_acc = evaluate(model, val_dataloader, loss_fn, device)
            print(f"Val Loss: {val_loss:.2f} - Val Acc: {val_acc:.2f}")
    save_info = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }    
    torch.save(save_info, f"checkpoint_one.pt")
    print(colored("Training Completed!", "blue"))

def evaluate(model, val_dataloader, loss_fn, device):
    model.eval()
    val_loss = []
    val_acc = []
    for batch in val_dataloader:
        sent, input_ids, attn_mask = batch.values()

        attn_mask = attn_mask.to(device)
        input_ids = input_ids.to(device)
        sent = sent.to(device)

        with torch.no_grad():
            logits = model(input_ids, attn_mask)
        
        loss = loss_fn(logits, sent)
        val_loss.append(loss)

        preds = torch.argmax(logits, dim=1).flatten()
        acc = (preds == sent).cpu().numpy().mean() * 100

        val_acc.append(acc)

    val_loss = np.mean(val_loss)
    val_acc = np.mean(val_acc)
    return val_loss, val_acc