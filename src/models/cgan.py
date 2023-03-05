from typing import Optional, Tuple
import torch
from torch import nn
import torch.utils.data
import torchvision
from torchvision import transforms

import pytorch_lightning as pl


# out = (in-1)*stride - 2*padding + (kernel_size-1) + 1
# output_size = input_size * 2
CONV_KWARGS_X2 = dict(
    kernel_size=4,
    stride=2,
    padding=1,
)

# output_size = input_size * 4
CONV_KWARGS_X4 = dict(
    kernel_size=6,
    stride=4,
    padding=1,
)


def generator_block(
    in_channels,
    out_channels,
    add_norm_and_activation=True,
    conv_kwargs=CONV_KWARGS_X2,
):
    layers = [
        nn.utils.spectral_norm(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                bias=False,
                **conv_kwargs,
            )
        ),
    ]
    if add_norm_and_activation:
        layers += [
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
    return layers


def discriminator_block(in_channels, out_channels, conv_kwargs=CONV_KWARGS_X2):
    # output_size = input_size * 4
    # out = (in-1)*stride - 2*padding + (kernel_size-1) + 1
    layers = [
        nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                bias=False,
                **conv_kwargs,
            ),
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
    ]
    return layers


class Generator(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()

        assert latent_size % 16 == 0

        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Linear(latent_size, 256 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(-1, (256, 4, 4)),
            *generator_block(256, 128),
            *generator_block(128, 64),
            *generator_block(64, 32),
            *generator_block(32, 16),
            *generator_block(16, 1, add_norm_and_activation=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        assert latent_size % 16 == 0
        self.latent_size = latent_size

        self.transform = nn.Sequential(
            transforms.GaussianBlur((5, 5), sigma=(1e-5, 0.5)),
            transforms.RandomResizedCrop((128, 128), scale=(0.8, 1), ratio=(0.8, 1.1))
        )
        self.main = nn.Sequential(
            *discriminator_block(1, 16),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.Flatten(),
        )
        self.linear1 = nn.Linear(256 * 4 * 4 + 1, 512)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, x, y):
        x = self.transform(x)
        x = self.main(x)
        x = self.linear1(torch.cat([x, y.view(-1, 1)], dim=-1))
        x = self.linear2(x)
        return x


class CGAN(pl.LightningModule):
    def __init__(
        self,
        latent_size: int = 256,
        lr: float = 3e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_size)
        self.discriminator = Discriminator(latent_size)
        self.fixed_noise_neg = self.generate_latent(9, y=torch.zeros(9))
        self.fixed_noise_pos = self.generate_latent(9, y=torch.ones(9))
        self.criterion = nn.BCEWithLogitsLoss()

    @property
    def device(self):
        return next(self.generator.parameters()).device

    def generate_latent(self, batch_size: int=1, y: Optional[torch.Tensor]=None):
        latent_size: int = self.hparams.latent_size # type: ignore
        latent = torch.randn(batch_size, latent_size-1)
        if y is None:
            y = torch.rand(batch_size) > 0.5
        return torch.cat([latent, y.view(-1, 1)], dim=1)

    def forward(self, noise):
        return self.generator(noise)

    def generate_images(self, batch_size: int, y: Optional[torch.Tensor]=None):
        noise = self.generate_latent(batch_size, y)
        return noise[:, -1].to(self.device), self(noise.to(self.device))

    def generator_loss(self, imgs: torch.Tensor, y: torch.Tensor, stage: str):
        fake_y, fake_imgs = self.generate_images(len(imgs))
        preds = self.discriminator(fake_imgs, fake_y)
        loss = self.criterion(preds.view(-1), torch.ones_like(fake_y))

        self.log(f"loss/generator/{stage}", loss, on_step=False, on_epoch=True)

        return loss

    def discriminator_loss(self, imgs: torch.Tensor, y: torch.Tensor, stage: str):
        y = y.to(torch.float32)
        
        # Compute discriminator loss on real data
        preds = self.discriminator(imgs, y)
        loss_real = self.criterion(preds.view(-1), torch.ones_like(y))
        pred_classes = preds > 0.5
        acc_real = (pred_classes == 1).float().mean()

        # Compute discriminator loss on fake data
        fake_y, fake_imgs = self.generate_images(len(imgs))
        preds = self.discriminator(fake_imgs.detach(), fake_y)
        loss_fake = self.criterion(preds.view(-1), torch.zeros_like(fake_y))
        pred_classes = preds > 0.5
        acc_fake = (pred_classes == 0).float().mean()

        # Overall loss
        loss = loss_real + loss_fake

        # Logging
        self.log_dict(
            {
                f"loss/discriminator/{stage}": loss,
                f"loss/discriminator_real/{stage}": loss_real,
                f"loss/discriminator_fake/{stage}": loss_fake,
                f"accuracy/real/{stage}": acc_real,
                f"accuracy/fake/{stage}": acc_fake,
            },
            on_step=False,
            on_epoch=True,
        )

        return loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ):
        if optimizer_idx == 0:
            # Optimize the generator
            self.generator.zero_grad()
            return self.generator_loss(*batch, stage="train")
        else:
            # Optimize the discriminator
            self.discriminator.zero_grad()
            return self.discriminator_loss(*batch, stage="train")

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        self.generator_loss(*batch, stage="val")
        self.discriminator_loss(*batch, stage="val")

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        self.generator_loss(*batch, stage="test")
        self.discriminator_loss(*batch, stage="test")

    def configure_optimizers(self):
        lr: float = self.hparams.lr  # type: ignore
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return opt_g, opt_d

    def _log_grid(self, name: str, latent: torch.Tensor) -> torch.Tensor:
        # Log images to tensorboard
        with torch.no_grad():
            imgs = self(latent.to(self.device))
        grid_size = int(len(latent) ** 0.5)
        grid = torchvision.utils.make_grid(imgs, nrow=grid_size, pad_value=2)
        self.logger.experiment.add_image(name, grid, self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        self._log_grid("fake_images/neg", self.fixed_noise_neg)
        self._log_grid("fake_images/pos", self.fixed_noise_pos)
