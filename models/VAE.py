import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from typing import Dict
import pprint
import pytorch_lightning as pl
import torch
import torch.optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from data import MNISTDataModule
from data import BreastCancerDataModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint



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


def conv_shape(input_size, kernel=1, stride=1, pad=0):
    from math import floor
    size = floor( ((input_size + 2*pad - kernel)/stride) + 1)
    return size

class VAE(pl.LightningModule):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config


        dense_layer_size = conv_shape(config['img_size'], config['kernel_size'], stride=2, pad=1)
        dense_layer_size = conv_shape(dense_layer_size, config['kernel_size'], stride=2)
        dense_layer_size = conv_shape(dense_layer_size, config['kernel_size'], stride=2)
        dense_layer_size = conv_shape(dense_layer_size, config['kernel_size'], stride=2)

        self.encoder = Encoder(config['latent_dimension'], config['kernel_size'], dense_layer_size, 1, config)
        self.decoder = Decoder(config['latent_dimension'], config['kernel_size'], dense_layer_size, config)
        
    
    def forward(self, x : torch.Tensor, labels: torch.Tensor):
        z = self.encoder(x, labels)
        y = self.decoder(z, labels)
        return y

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs, labels)
        loss = ((outputs - inputs)**2).mean() + self.encoder.kl

        self.log_dict({
            'loss': loss,
            'kl': self.encoder.kl,
            'recon_loss': ((outputs - inputs)**2).mean(),
            
        })
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch 
        inputs = inputs.to(self.config['device'])

        outputs = self.forward(inputs, labels)

        loss = ((outputs - inputs)**2).mean()
        self.log_dict({
            'loss': loss,
            'kl_val': self.encoder.kl,
            'recon_loss_val': ((outputs - inputs)**2).mean()
        })

        if batch_idx == 0:

            grid = torchvision.utils.make_grid([inputs[0, :, :], outputs[0, :, :]], nrow=1, pad_value=1)
            self.logger.experiment.add_image("reconstruction", grid, self.current_epoch)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['lr'])





class Encoder(nn.Module):
    def __init__(self, latent_dim, kernel_size, size, in_channels, config) -> None:
        super(Encoder, self).__init__()
        self.c1 = config['c1']
        self.c2 = config['c2']
        self.c3 = config['c3']
        self.c4 = config['c4']

        self.conv1 = nn.Conv2d(in_channels, self.c1, kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.c1, self.c2, kernel_size, stride=2)
        self.conv3 = nn.Conv2d(self.c2, self.c3, kernel_size, stride=2)
        self.conv4 = nn.Conv2d(self.c3, self.c4, kernel_size, stride=2)

        self.size_fc = self.c4*size*size

        
        self.linear1 = nn.Linear(self.size_fc + 1, config['fc_size'])
        self.fc = nn.Linear(config['fc_size'], latent_dim)

        self.fc_mu = nn.Linear(config['fc_size'], latent_dim)
        self.fc_sigma = nn.Linear(config['fc_size'], latent_dim)

        self.dropout = nn.Dropout(p=config['dropout_p'], inplace=True)
        self.batchnorm = nn.BatchNorm2d(self.c2)


        if config['activation'] == 'relu':
            self.activation = nn.ReLU(True)
        else:
            self.activation = nn.Sigmoid()

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x : torch.Tensor, labels : torch.Tensor):
        x = self.conv1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.activation(x)

        x = self.conv4(x)
        x = self.activation(x)


        x = x.view(-1, self.size_fc)

        #concat label
        labels = labels.unsqueeze(-1)
        x = torch.cat((x, labels), 1)
        
        x = self.linear1(x)
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))

        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma**2) - 1).sum()

        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, kernel_size, deconv1_size, config):
        super(Decoder, self).__init__()
        self.c1 = config['c1']
        self.c2 = config['c2']
        self.c3 = config['c3']
        self.c4 = config['c4']

        self.deconv1_size = deconv1_size
        self.first_layer = nn.Linear(latent_dim + 1, config['fc_size'])
        self.linear2 = nn.Linear(2048, self.c4*deconv1_size*deconv1_size)
        self.deconv1 = nn.ConvTranspose2d(self.c4, self.c3, kernel_size, stride=2)  
        self.deconv2 = nn.ConvTranspose2d(self.c3, self.c2, kernel_size, stride=2)
        self.deconv3 = nn.ConvTranspose2d(self.c2, self.c1, kernel_size, stride=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(self.c1, 1, kernel_size, stride=2)

        self.batchnorm = nn.BatchNorm2d(self.c2)
        self.dropout = nn.Dropout(p=config['dropout_p'], inplace=True)
        

        if config['activation'] == 'relu':
            self.activation = nn.ReLU(True)
        else:
            self.activation = nn.Sigmoid()
    
    def forward(self, x : torch.Tensor, labels : torch.Tensor):

        #concat label
        labels = labels.unsqueeze(-1)
        x = torch.cat((x, labels), 1)

        x = self.first_layer(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.activation(x)


        x = x.view(-1, self.c4, self.deconv1_size, self.deconv1_size)
        x = self.deconv1(x)
        x = self.activation(x)

        
        x = self.deconv2(x)
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.activation(x)

        x = self.deconv3(x)
        x = self.activation(x)

        x = self.deconv4(x)

        return x

    
def main():

    print('Starting training: ')
    pprint.pprint(config)

    pl.seed_everything(1234)
    vae = VAE(config)
    vae = vae.to(config["device"])

    cancer_dataset = BreastCancerDataModule(preload=True, num_workers=config['num_workers'])
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