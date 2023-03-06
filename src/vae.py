import torch
from torch import nn
import pytorch_lightning as pl

class SamplingPosterior(nn.Module):
    def __init__(self, config, debug=False):
        super(SamplingPosterior, self).__init__()
        self.config = config
    
    def forward(self, mu, logstd):

        std =  torch.exp(logstd)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return z


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()

        self.config = config
        self.encoder_layer =  nn.Sequential(
            nn.Conv2d(1, self.config['f1'], kernel_size=(self.config['n1'],self.config['m1'])),
            nn.BatchNorm2d(self.config['f1']),
            nn.ELU(),
            nn.Conv2d(self.config['f1'],self.config['f2'], kernel_size=(self.config['n2'],self.config['m2'])),
            nn.BatchNorm2d(self.config['f2']),
            nn.ELU(),
            nn.Conv2d(self.config['f2'],self.config['f3'], kernel_size=(self.config['n3'], self.config['m3'])),
            nn.BatchNorm2d(self.config['f3']),
            nn.ELU(),
            nn.Flatten(),
        )

    def forward(self,input: torch.DoubleTensor):
        input = input.view(input.shape[0], 1, self.config['img_size'], self.config['img_size'])
        return self.encoder_layer(input)

class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.decoder_layer =  nn.Sequential(
            nn.Linear(self.config['latent_dimension'],self.config['f3']*(self.config['img_size'] - self.config['m1'] - self.config['m2'] -self.config['m3']+3)**2),
            nn.Unflatten(1,(self.config['f3'],  (self.config['img_size'] - self.config['m1'] - self.config['m2'] - self.config['m3'] + 3), (self.config['img_size'] - self.config['m1'] - self.config['m2'] - self.config['m3'] + 3))),
            nn.ConvTranspose2d(self.config['f3'],self.config['f2'], kernel_size=(self.config['n3'], self.config['m3'])),
            nn.BatchNorm2d(self.config['f2']),
            nn.ELU(),
            nn.ConvTranspose2d(self.config['f2'],self.config['f1'], kernel_size=(self.config['n2'],self.config['m2'])),
            nn.BatchNorm2d(self.config['f1']),
            nn.ELU(),
            nn.ConvTranspose2d(self.config['f1'],1, kernel_size=(self.config['n1'],self.config['m1'])),
        )

    def forward(self,input):
        return self.decoder_layer(input)

class VariationalAutoencoder(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config 

        self.encoder_layer = Encoder(config)
        self.decoder_layer = Decoder(config)
        self.sampling_layer = SamplingPosterior(config)

        self.mu = nn.Linear(self.config['f3']*((self.config['img_size'] - self.config['m1'] - self.config['m2'] - self.config['m3'] + 3)**2), self.config['latent_dimension'])
        self.logstd = nn.Linear(self.config['f3']*((self.config['img_size'] - self.config['m1'] - self.config['m2'] - self.config['m3'] + 3)**2), self.config['latent_dimension'])

    def forward(self, input):
        input = self.encoder_layer(input)
        mu = self.mu(input)
        logstd = self.logstd(input)

        z = self.sampling_layer(mu, logstd)

        x_decoded = self.decoder_layer(z)
        #x_decoded = x_decoded.view(x_decoded.shape[0], self.config['dim'], self.config['img_size'])

        return x_decoded, z    
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_encoded = self.encoder_layer(x)
        mu = self.mu(x_encoded)
        logstd = self.logstd(x_encoded)

        z = self.sampling_layer(mu, logstd)

        x_decoded = self.decoder_layer(z)
        self.logger.experiment.add_image("fake_images", x_decoded[0, :, :, :], self.current_epoch)


    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_encoded = self.encoder_layer(x)
        mu = self.mu(x_encoded)
        logstd = self.logstd(x_encoded)

        z = self.sampling_layer(mu, logstd)

        x_decoded = self.decoder_layer(z)
        #x_decoded = x_decoded.view(x_decoded.shape[0], self.config['dim'], self.config['img_size'])

        elbo = self.loss(mu, logstd, x_decoded, x)

        return elbo

    def loss(self,mu, logstd, x_decoded, x):
        std = torch.exp(logstd)
        kl_loss = -0.5 * torch.sum(1 + std - mu**2 - torch.exp(std)) 
        pred_loss = nn.MSELoss(reduction='mean')(x,x_decoded)

        elbo = kl_loss + pred_loss
        elbo = elbo.mean()

        self.log_dict({
                    'elbo': elbo,
                    'kl': kl_loss.mean(),
                    'recon_loss': pred_loss.mean(),
                })

        return elbo

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.config['lr'],weight_decay=1e-5)
    