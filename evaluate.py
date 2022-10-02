import config
import torch
from dataset import eval_dataloader, test_dataloader
from pytorch_lightning import Trainer, seed_everything
# from sklearn.metrics import classification_report
from trainer import PhoBERTTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PhoBERTTrainer()
model.to(device)
checkpoint = torch.load(config.CKPT_PATH)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
model.freeze()

if __name__ == "__main__":
    seed_everything(69)

    trainer = Trainer(accelerator='gpu')

    print("\nEvaluate on Validation Set:\n")
    trainer.validate(model=model, ckpt_path=config.CKPT_PATH, dataloaders=eval_dataloader)

    print("\nEvaluate on Test Set:\n")
    trainer.test(model=model, ckpt_path=config.CKPT_PATH, dataloaders=test_dataloader)