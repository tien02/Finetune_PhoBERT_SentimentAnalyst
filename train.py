import config.train_config as train_config
from dataset import train_dataloader, eval_dataloader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from trainer import PhoBERTModel
from pytorch_lightning.loggers import TensorBoardLogger
from model import PhoBERTLSTM_base, PhoBERTLSTM_large, PhoBertFeedForward_base, PhoBertFeedForward_large
from termcolor import colored

def main():
        seed_everything(69)
        
        if train_config.MODEL == "FeedForward-base":
                model = PhoBertFeedForward_base()
                print(colored("\nUse PhoBERT FeedForward base\n", "green"))
        elif train_config.MODEL == "FeedForward-large":
                model = PhoBertFeedForward_large()
                print(colored("\nUse PhoBERT FeedForward large\n", "green"))
        elif train_config.MODEL == "LSTM-base":
                model = PhoBERTLSTM_base()
                print(colored("\nUse PhoBERT LSTM base\n", "green"))
        else:
                model = PhoBERTLSTM_large()
                print(colored("\nUse PhoBERT LSTM large\n", "green"))
        system = PhoBERTModel(model=model)

        checkpoint_callback = ModelCheckpoint(dirpath= train_config.CKPT_DIR, monitor="val_loss",
                                                save_top_k=3, mode="min")
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", check_finite=True)

        logger = TensorBoardLogger(train_config.CKPT_DIR, name=train_config.LOGGER["name"], version=train_config.LOGGER["version"])

        trainer = Trainer(accelerator='gpu', check_val_every_n_epoch=train_config.VAL_EACH_EPOCH,
                        gradient_clip_val=1.0,max_epochs=train_config.EPOCHS,
                        enable_checkpointing=True, deterministic=True, default_root_dir=train_config.CKPT_DIR,
                        callbacks=[checkpoint_callback, early_stopping], logger=logger)
        trainer.fit(model=system, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=eval_dataloader, 
                ckpt_path=train_config.CKPT_RESUME_TRAIN)

if __name__ == "__main__":
        main()