import config
from dataset import train_dataloader, eval_dataloader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from trainer import PhoBERTModel
from pytorch_lightning.loggers import TensorBoardLogger
from model import PhoBertFeedForward, PhoBERTLSTM
from termcolor import colored

def main():
        seed_everything(69)
        
        if config.MODEL == "FeedForward":
                model = PhoBertFeedForward()
                print(colored("\nUse PhoBERT FeedForward Network\n", "green"))
        elif config.MODEL == "LSTM":
                model = PhoBERTLSTM()
                print(colored("\nUse PhoBERT LSTM Network\n", "green"))
        else:
                print(colored("\nUse PhoBERT CNN Network\n", "green"))
                pass
        system = PhoBERTModel(model=model)

        checkpoint_callback = ModelCheckpoint(dirpath= config.CKPT_DIR, monitor="val_loss",
                                                save_top_k=3, mode="min")
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", check_finite=True)

        logger = TensorBoardLogger(config.CKPT_DIR, name=config.LOGGER["name"], version=config.LOGGER["version"])

        trainer = Trainer(accelerator='gpu', check_val_every_n_epoch=config.VAL_EACH_EPOCH,
                        gradient_clip_val=1.0,max_epochs=config.EPOCHS,
                        enable_checkpointing=True, deterministic=True, default_root_dir=config.CKPT_DIR,
                        callbacks=[checkpoint_callback, early_stopping], logger=logger)
        trainer.fit(model=system, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=eval_dataloader, 
                ckpt_path=config.CKPT_RESUME_TRAIN)

if __name__ == "__main__":
        main()
