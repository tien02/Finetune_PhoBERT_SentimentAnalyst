import config
from dataset import train_dataloader, eval_dataloader
from pytorch_lightning import Trainer, seed_everything
from trainer import PhoBERTTrainer

def main():
        seed_everything(69)

        model = PhoBERTTrainer()
        trainer = Trainer(accelerator='gpu', check_val_every_n_epoch=config.VAL_EACH_EPOCH,
                        gradient_clip_val=1.0,max_epochs=config.EPOCHS,
                        enable_checkpointing=True, deterministic=True)
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=eval_dataloader)

if __name__ == "__main__":
        main()
