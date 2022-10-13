import config.train_config as train_config
import torch
from dataset import eval_dataloader, test_dataloader
from pytorch_lightning import Trainer, seed_everything
from model import PhoBERTLSTM_base, PhoBERTLSTM_large, PhoBertFeedForward_base, PhoBertFeedForward_large
from trainer import PhoBERTModel
from termcolor import colored

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if train_config.MODEL == "FeedForward-base":
    model = PhoBertFeedForward_base()
    print(colored("\nEvaluate PhoBERT FeedForward base\n", "green"))
elif train_config.MODEL == "FeedForward-large":
    model = PhoBertFeedForward_large()
    print(colored("\nEvaluate PhoBERT FeedForward large\n", "green"))
elif train_config.MODEL == "LSTM-base":
    model = PhoBERTLSTM_base()
    print(colored("\nEvaluate PhoBERT LSTM base\n", "green"))
else:
    model = PhoBERTLSTM_large()
    print(colored("\nEvaluate PhoBERT LSTM large\n", "green"))
system = PhoBERTModel(model)
system.to(device)
checkpoint = torch.load(train_config.CKPT_PATH)
system.load_state_dict(checkpoint["state_dict"])
system.eval()
system.freeze()

if __name__ == "__main__":
    seed_everything(69)

    trainer = Trainer(accelerator='gpu')

    print("\nEvaluate on Validation Set:\n")
    trainer.validate(model=system, ckpt_path=train_config.CKPT_PATH, dataloaders=eval_dataloader)

    print("\nEvaluate on Test Set:\n")
    trainer.test(model=system, ckpt_path=train_config.CKPT_PATH, dataloaders=test_dataloader)