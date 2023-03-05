import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pprint
import pytorch_lightning as pl
import torch
import torch.optim
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import MNISTDataModule
from data import BreastCancerDataset128
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def conv_shape(input_size, kernel=1, stride=1, pad=0):
    from math import floor
    size = floor( ((input_size + 2*pad - kernel)/stride) + 1)
    return size


class VAE(pl.LightningModule):
    def __init__(self, latent_dim, img_size, config):
        super(VAE, self).__init__()
        self.config = config
        dense_layer_size = conv_shape(img_size, kernel=3, stride=1)
        dense_layer_size = conv_shape(dense_layer_size, kernel=3, stride=1)
        dense_layer_size = conv_shape(dense_layer_size, kernel=3, stride=2)
        dense_layer_size = conv_shape(dense_layer_size, kernel=3, stride=2)

        self.decoder = Decoder(latent_dim, 3, dense_layer_size, config)
        self.encoder = Encoder(1, 3, latent_dim, dense_layer_size, config)
    
    def forward(self, x : torch.Tensor, labels: torch.Tensor):
        z = self.encoder(x, labels)
        y = self.decoder(z, labels)
        return y

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs, labels)
        loss = ((outputs - inputs)**2).mean()

        self.log_dict({
            'elbo': loss,
            'kl': self.encoder.kl,
            'recon_loss': ((outputs - inputs)**2).mean(),
            
        })
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch 
        inputs = inputs.to(device)

        outputs = self.forward(inputs, labels)

        loss = ((outputs - inputs)**2).sum() + self.encoder.kl
        self.log_dict({
            'elbo_val': loss,
            'kl_val': self.encoder.kl,
            'recon_loss_val': ((outputs - inputs)**2).sum()
        })

        if batch_idx == 0:
            # img = plot_reconstructed(self)
            # self.logger.experiment.add_image("latent_generation", img, self.current_epoch)

            grid = torchvision.utils.make_grid([inputs[0, :, :], outputs[0, :, :]], nrow=1, pad_value=1)
            self.logger.experiment.add_image("reconstruction", grid, self.current_epoch)



    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class Encoder(nn.Module):
    def __init__(self, in_channels, kernel_size, latent_dim, size, config) -> None:
        super(Encoder, self).__init__()
        c1 = 8
        c2 = 16
        c3 = 32
        c4 = 32

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size, stride=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size, stride=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size, stride=2)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size, stride=2)

        
        self.batchnorm = nn.BatchNorm2d(c2)


        self.size_fc = c4*size*size

        
        self.linear1 = nn.Linear(self.size_fc + 1, 128)

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_sigma = nn.Linear(128, latent_dim)

        self.dropout = nn.Dropout(p=0.05, inplace=True)
        self.relu = nn.ReLU()


        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x : torch.Tensor, labels : torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        #x = self.batchnorm(x)
        #x = self.dropout(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        
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
        self.c1 = 8
        self.c2 = 16
        self.c3 = 32
        self.c4 = 32

        self.deconv1_size = deconv1_size
        self.first_layer = nn.Linear(latent_dim + 1, 128)
        self.linear2 = nn.Linear(128, self.c4*deconv1_size*deconv1_size)
        self.deconv1 = nn.ConvTranspose2d(self.c4, self.c3, kernel_size, stride=2)  
        self.deconv2 = nn.ConvTranspose2d(self.c3, self.c2, kernel_size, stride=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(self.c2, self.c1, kernel_size, stride=1)
        self.deconv4 = nn.ConvTranspose2d(self.c1, 1, kernel_size, stride=1)
        
        #self.batchnorm = nn.BatchNorm2d(self.c2)
        #self.dropout = nn.Dropout(p=0.05, inplace=True)
        self.relu = nn.ReLU()
    
    def forward(self, x : torch.Tensor, labels : torch.Tensor):

        #concat label
        labels = labels.unsqueeze(-1)
        x = torch.cat((x, labels), 1)

        x = self.first_layer(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)


        x = x.view(-1, self.c4, self.deconv1_size, self.deconv1_size)
        x = self.deconv1(x)
        x = self.relu(x)

        
        x = self.deconv2(x)
        #x = self.dropout(x)
        #x = self.batchnorm(x)
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
                    loss = ((outputs - inputs)**2).sum() + model.encoder.kl

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

    return

        
def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_reconstructed(autoencoder : VAE, r0=(-5, 10), r1=(-10, 5), n=10):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    return img



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
    "device": "cuda:0" if torch.cuda.is_available() else "cpu", 
    "img_size": 28,
    "num_workers": 8,
    "batch_size": 32,
    'lr': 1e-3,
    "max_epochs": 50,
}


def main():
    
    # minist_dataset = torchvision.datasets.MNIST(root='data', download=True, train=True, transform=torchvision.transforms.ToTensor())
    # dataset_train, dataset_test  = random_split(minist_dataset, [int(0.7*len(minist_dataset)), len(minist_dataset) - int(0.7*len(minist_dataset))])
    
    # dataloaders = {}
    # dataloaders['train'] = DataLoader(dataset_train, batch_size=512, num_workers=4)
    # dataloaders['val'] = DataLoader(dataset_test, batch_size=512, num_workers=4)

    # model = VAE(img_size=28, latent_dim=2)

    # model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train(model, optimizer, None, dataloaders, 2)        

    # model.eval()


    print('Starting training: ')
    pprint.pprint(config)

    pl.seed_everything(1234)
    vae = VAE(56, 28, config)
    #summary(vae, (1, config['batch_size'] ,1, config['img_size'], config['img_size']))
    vae = vae.to(config["device"])

    mnist = MNISTDataModule(batch_size=config["batch_size"], num_workers=config["num_workers"], img_size=config["img_size"], only_zero_one=True) 
    cancer_dataset = BreastCancerDataset128()
    vit_callback = ModelCheckpoint(monitor=r'elbo',mode='min')

    trainer = pl.Trainer(
        accelerator= "gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=config["max_epochs"],
        log_every_n_steps=10,
        callbacks=[vit_callback]
    )
    # vae = VAE.load_from_checkpoint('./lightning_logs/version_19/checkpoints/epoch=49-step=1000.ckpt', latent_dim=56, img_size=28, config=config)
    # vae.to(device)
    # vae.eval()
    trainer.fit(vae, mnist)

    # img = plot_reconstructed(vae, r0=(-5, 10), r1=(-10, 5))
    # plt.imshow(img)
    # plt.show()


    latent = torch.randn(128, 56, device=device)
    img_recon = vae.decoder(latent)
    img_recon = img_recon.cpu()

    fig, ax = plt.subplots(figsize=(20, 8.5))
    
    show_image(torchvision.utils.make_grid(img_recon.data[:100],10,5))
    plt.show()
    


if __name__ == "__main__":
    main()