import config
import torch
from dataset import eval_dataloader, test_dataloader
from pytorch_lightning import Trainer, seed_everything
from model import PhoBertFeedForward, PhoBERTLSTM
from trainer import PhoBERTModel
from termcolor import colored

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if config.MODEL == "FeedForward":
    model = PhoBertFeedForward(from_pretrained=False)
    print(colored("\Evaluate PhoBERT FeedForward Network\n", "green"))
elif config.MODEL == "LSTM":
    model = PhoBERTLSTM(from_pretrained=False)
    print(colored("\nEvaluate PhoBERT LSTM Network\n", "green"))
else:
    print(colored("\nEvaluate PhoBERT CNN Network\n", "green"))
    pass
system = PhoBERTModel()
system.to(device)
checkpoint = torch.load(config.CKPT_PATH)
system.load_state_dict(checkpoint["state_dict"])
system.eval()
system.freeze()

if __name__ == "__main__":
    seed_everything(69)

    trainer = Trainer(accelerator='gpu')

    print("\nEvaluate on Validation Set:\n")
    trainer.validate(model=system, ckpt_path=config.CKPT_PATH, dataloaders=eval_dataloader)

    print("\nEvaluate on Test Set:\n")
    trainer.test(model=system, ckpt_path=config.CKPT_PATH, dataloaders=test_dataloader)