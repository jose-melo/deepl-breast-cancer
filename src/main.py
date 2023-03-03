import argparse
from typing import Tuple
import torch
import pytorch_lightning
from torchvision import datasets
from torchvision import transforms
import os
from torch.utils.data import DataLoader
from torch import nn

from models import AttentionLayer, EmbeddingLayer, ViT

defaut_config = {
    "img_size": 28,
    "data_path": "data",
    "batch_size": 32,
    "num_workers": 1,
    "epochs": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu", 
    "n_channels": 1,
    "embedding_dim": 2, 
    "head_embedding_dim": 2, 
    "patch": 7,
    "dim_feed_forward": 16,
    "num_classes": 2,
    "num_layers": 4,
    "lr": 1e-4
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
    
        return train_dataloader, test_dataloader


    def train(self):
        iter_per_epoch = len(self.train_loader)

        optimizer = torch.optim.AdamW(self.model.parameters(), self.config["lr"], weight_decay=1e-3)
        cos_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config["epochs"])
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(self.config["epochs"]):
            for i, (imgs, labels) in enumerate(self.train_loader):

                imgs, labels = imgs.to(self.config["device"]), labels.to(self.config["device"])

                predict = self.model(imgs)
                loss = loss_fn(predict, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            cos_decay.step()



if __name__ == '__main__':

    vit = ViTMNIST(defaut_config)
    vit.get_data()
    vit.train()


    

