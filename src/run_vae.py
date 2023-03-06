from typing import Dict
import pytorch_lightning as pl
import torch
import torch.optim
from data import BreastCancerDataModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pprint
from models.vae import VAE


config = {
    'c1': 8,
    'c2': 16,
    'c3': 32,
    'c4': 64,
    'kernel_size': 4,
    'latent_dimension': 64,
    "img_size": 128,
    "fc_size": 2048,
    "dropout_p": 0.05,
    "activation": 'relu',
    "num_workers": 0,
    "batch_size": 64,
    'lr': 1e-5,
    "max_epochs": 50,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu"
}


def main():

    print('Starting training: ')
    pprint.pprint(config)

    pl.seed_everything(1234)
    vae = VAE(config)
    vae = vae.to(config["device"])

    cancer_dataset = BreastCancerDataModule(preload=False, num_workers=config['num_workers'])
    vit_callback = ModelCheckpoint(monitor=r'loss',mode='min')

    trainer = pl.Trainer(
        accelerator= "gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=config["max_epochs"],
        log_every_n_steps=10,
        callbacks=[vit_callback]
    )
    trainer.fit(vae, cancer_dataset)


if __name__ == "__main__":
    main()