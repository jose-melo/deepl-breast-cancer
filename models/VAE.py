import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch
import torch.optim
import torchvision
import numpy as np
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

size_fc = 5 * 5* 5
class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 5, 3)

        self.fc = nn.Linear(size_fc, 10)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x : torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.relu(x)

        x = x.view(-1, size_fc)

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

    model = Encoder()
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    loss = torch.nn.CrossEntropyLoss()

    train(model, optimizer, loss, dataloaders, 100)        


if __name__ == "__main__":
    main()