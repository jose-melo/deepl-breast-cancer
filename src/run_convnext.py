import torch
from models import ViTMNIST
import pprint
from torchinfo import summary
from convnext import ConvNextSmall
import pytorch_lightning as pl
from data import BreastCancerDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

config = {
    "img_size": 128,
    "data_path": "data",
    "batch_size": 64,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu", 
    "n_channels": 1,
    "num_classes": 2,
    "lr": 1e-4,
    "max_epochs": 50,
}

if __name__ == '__main__':

    print('Starting training: ')
    pprint.pprint(config)
    pl.seed_everything(1234)
    
    model = ConvNextSmall(config) 
    summary(model, (config["img_size"], config["n_channels"], config["img_size"], config["img_size"]))
    model = model.to(config["device"])

    data = BreastCancerDataModule(batch_size=config["batch_size"], num_workers=config["num_workers"], preload=config['device'] == 'cuda', augment=True, load_extra_from='data/train_images_gen')        

    resnet_callback = ModelCheckpoint(monitor=r'val_loss',mode='min')
    trainer = pl.Trainer(
        accelerator= "gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=config["max_epochs"],
        log_every_n_steps=1,
        callbacks=[resnet_callback]
    )

    trainer.fit(model, data)

    

