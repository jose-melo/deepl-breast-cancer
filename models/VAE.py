import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch
import torch.optim
import torchvision
import numpy as np
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    def __init__(self, img_size, in_channels, kernel_size=3) -> None:
        super(Encoder, self).__init__()
        conv1_out = 10
        conv2_out = 10
        maxpool_size = 2
        maxpool_stride = 2

        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size)

        img_size = img_size - (kernel_size) + 1 # conv1
        img_size = img_size // 2 # maxpool1
        img_size = img_size - (kernel_size) + 1 # conv2
        img_size = img_size // 2 # maxpool2
        self.size_fc = img_size*img_size*conv2_out

        self.fc = nn.Linear(self.size_fc, 10)

        self.maxpool = nn.MaxPool2d(maxpool_size, maxpool_stride)
        self.relu = nn.ReLU()

    def forward(self, x : torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = x.view(-1, self.size_fc)

        x = self.fc(x)

        return x


def train(model : nn.Module, optimizer : torch.optim.Optimizer, criterion : torch.nn.Module, dataloaders : DataLoader, num_epochs=100):
    for epochs in range(num_epochs):
        for mode in ['train', 'val']:
            if(mode == 'val'):
                model.eval()
            else:
                model.train()
            
            running_loss = 0.0
            running_acc = 0

            for inputs, labels in dataloaders[mode]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(mode == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloaders[mode].dataset)
            epoch_acc = running_acc / len(dataloaders[mode].dataset)

            print(f"[{mode}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")


            
def main():
    
    minist_dataset = torchvision.datasets.MNIST(root='data', download=True, train=True, transform=torchvision.transforms.ToTensor())
    dataset_train, dataset_test  = random_split(minist_dataset, [int(0.7*len(minist_dataset)), len(minist_dataset) - int(0.7*len(minist_dataset))])
    
    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset_train, batch_size=64)
    dataloaders['val'] = DataLoader(dataset_test, batch_size=64)

    model = Encoder(img_size=28, in_channels=1)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    loss = torch.nn.CrossEntropyLoss()

    train(model, optimizer, loss, dataloaders, 100)        


if __name__ == "__main__":
    main()