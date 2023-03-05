import torch
from data import MNISTDataModule
from models import ViTMNIST
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pprint
from torchinfo import summary

#from vae import VariationalAutoencoder
from vae_victor import VAE
import pytorch_lightning as pl


config = {
    'n1': 4,
    'm1': 4,
    'f1': 8,
    'n2': 4,
    'm2': 4,
    'f2': 16,
    'n3': 4,
    'm3': 4, 
    'f3': 32,
    'dim': 1,
    'latent_dimension': 2,
    "data_path": "data",
    "device": "cuda" if torch.cuda.is_available() else "cpu", 
    "img_size": 28,
    "num_workers": 8,
    "batch_size": 16,
    'lr': 1e-4,
    "max_epochs": 50,
}

if __name__ == '__main__':

    print('Starting training: ')
    pprint.pprint(config)

    pl.seed_everything(1234)
    vae = VAE(2, 28, config)
    #summary(vae, (1, config['batch_size'] ,1, config['img_size'], config['img_size']))
    vae = vae.to(config["device"])

    mnist = MNISTDataModule(batch_size=config["batch_size"], num_workers=config["num_workers"], img_size=config["img_size"])        

    vit_callback = ModelCheckpoint(monitor=r'elbo',mode='min')
    trainer = pl.Trainer(
        accelerator= "gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=config["max_epochs"],
        log_every_n_steps=1,
        callbacks=[vit_callback]
    )

    trainer.fit(vae, mnist)


    

