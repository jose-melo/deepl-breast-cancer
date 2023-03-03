from typing import Tuple
import torch
import pytorch_lightning as pl
from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from models import ViT

defaut_config = {
    "img_size": 28,
    "data_path": "data",
    "batch_size": 128,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu", 
    "n_channels": 1,
    "embedding_dim": 64, 
    "patch": 4,
    "dim_feed_forward": 128,
    "num_classes": 10,
    "num_layers": 6,
    "lr": 1e-4,
    "max_epochs": 50,
    "num_attention_heads": 4,
}

class ViTMNIST(object):

    def __init__(self, config: dict):
        self.config = config
        self.model = ViT(config)
        
    def get_data(self) -> Tuple[DataLoader, DataLoader]:
        transform_train = transforms.Compose([transforms.RandomCrop(self.config["img_size"], padding=2), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])

        train = datasets.MNIST(self.config["data_path"], train=True, download=True, transform=transform_train)

        transform_test = transforms.Compose([transforms.Resize([self.config["img_size"], self.config["img_size"]]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        test = datasets.MNIST(self.config["data_path"], train=False, download=True, transform=transform_test)
    
        train_dataloader = DataLoader(dataset=train,
                                    batch_size=self.config["batch_size"],
                                    shuffle=True,
                                    num_workers=self.config["num_workers"],
                                    drop_last=True)

        test_dataloader = DataLoader(dataset=test,
                                    batch_size=self.config["batch_size"],
                                    shuffle=False,
                                    num_workers=self.config["num_workers"],
                                    drop_last=False)

        self.train_loader = train_dataloader
        self.test_loader = test_dataloader

    def train(self):
        self.get_data()
        vit_callback = ModelCheckpoint(monitor=r'val_loss',mode='min')
        self.trainer = pl.Trainer(
            accelerator= "gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=self.config["max_epochs"],
            log_every_n_steps=1,
            callbacks=[vit_callback]
        )

        self.trainer.fit(
            model=self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.test_loader
        )

        self.trainer.save_checkpoint('ckpt_save.ckpt')        

if __name__ == '__main__':

    vit = ViTMNIST(defaut_config)
    vit.train()


    

