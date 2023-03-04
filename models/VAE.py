import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch
import torch.optim
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def conv_shape(input_size, kernel=1, stride=1, pad=0):
    from math import floor
    size = floor( ((input_size + 2*pad - kernel)/stride) + 1)
    return size


class AE(nn.Module):
    def __init__(self, latent_dim, img_size):
        super(AE, self).__init__()
        dense_layer_size = conv_shape(img_size, kernel=3, stride=1)
        dense_layer_size = conv_shape(dense_layer_size, kernel=3, stride=1)
        dense_layer_size = conv_shape(dense_layer_size, kernel=3, stride=2)
        dense_layer_size = conv_shape(dense_layer_size, kernel=3, stride=2)

        self.decoder = Decoder(latent_dim, 3, dense_layer_size)
        self.encoder = Encoder(1, 3, latent_dim, dense_layer_size)
    
    def forward(self, x : torch.Tensor):
        z = self.encoder(x)
        y = self.decoder(z)

        return y


class Encoder(nn.Module):
    def __init__(self, in_channels, kernel_size, latent_dim, size) -> None:
        super(Encoder, self).__init__()
        c1 = 16
        c2 = 16
        c3 = 32
        c4 = 32

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size, stride=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size, stride=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size, stride=2)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size, stride=2)

        
        self.batchnorm = nn.BatchNorm2d(c2)
        
        

        self.size_fc = c4*size*size

        
        self.linear1 = nn.Linear(self.size_fc, 128)
        self.fc = nn.Linear(128, latent_dim)

        self.dropout = nn.Dropout(p=0.05, inplace=True)
        self.relu = nn.ReLU()

    def forward(self, x : torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        
        x = x.view(-1, self.size_fc)

        x = self.linear1(x)
        x = self.fc(x)

        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, kernel_size, deconv1_size):
        super(Decoder, self).__init__()
        self.c1 = 16
        self.c2 = 16
        self.c3 = 32
        self.c4 = 32

        self.deconv1_size = deconv1_size
        self.first_layer = nn.Linear(latent_dim, 128)
        self.linear2 = nn.Linear(128, self.c4*deconv1_size*deconv1_size)
        self.deconv1 = nn.ConvTranspose2d(self.c4, self.c3, kernel_size, stride=2)
        self.deconv2 = nn.ConvTranspose2d(self.c3, self.c2, kernel_size, stride=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(self.c2, self.c1, kernel_size, stride=1)
        self.deconv4 = nn.ConvTranspose2d(self.c1, 1, kernel_size, stride=1)
        
        self.batchnorm = nn.BatchNorm2d(self.c2)
        self.dropout = nn.Dropout(p=0.05, inplace=True)
        self.relu = nn.ReLU()
    
    def forward(self, x : torch.Tensor):
        
        x = self.first_layer(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)


        x = x.view(-1, self.c4, self.deconv1_size, self.deconv1_size)
        x = self.deconv1(x)
        x = self.relu(x)

        
        x = self.deconv2(x)
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        
        return x

def plot_img(inputs : torch.Tensor, outputs : torch.Tensor):
    inputs = inputs.detach().numpy()
    outputs = outputs.detach().numpy()
    inputs = inputs.squeeze(1)
    outputs = outputs.squeeze(1)

    plt.subplot(1, 2, 1)
    plt.imshow(inputs[0], label="input")

    plt.subplot(1, 2, 2)
    plt.imshow(outputs[0], label="output")
    plt.show()
    return
    


def train(model : nn.Module, optimizer : torch.optim.Optimizer, criterion : torch.nn.Module, dataloaders : dict[str,DataLoader], num_epochs=100):
    for epoch in range(num_epochs):
        for mode in ['train', 'val']:
            if(mode == 'val'):
                model.eval()
            else:
                model.train()
            
            running_loss = 0.0

            for inputs, _ in dataloaders[mode]:
                inputs = inputs.to(device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(mode == 'train'):
                    outputs = model(inputs)
                    loss = criterion(inputs, outputs)

                    if mode == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloaders[mode].dataset)

            print(f"[{mode}] Loss: {epoch_loss:.4f}")

        if epoch % 10 == 0:
            inputs = dataloaders['val'].dataset[0][0]
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(device)
            outputs = model(inputs)
            plot_img(inputs.cpu(), outputs.cpu())

            

        


            
def main():
    
    minist_dataset = torchvision.datasets.MNIST(root='data', download=True, train=True, transform=torchvision.transforms.ToTensor())
    dataset_train, dataset_test  = random_split(minist_dataset, [int(0.7*len(minist_dataset)), len(minist_dataset) - int(0.7*len(minist_dataset))])
    
    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset_train, batch_size=64, num_workers=4)
    dataloaders['val'] = DataLoader(dataset_test, batch_size=64, num_workers=4)

    model = AE(img_size=28, latent_dim=28)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.MSELoss()

    train(model, optimizer, loss, dataloaders, 100)        
    


if __name__ == "__main__":
    main()