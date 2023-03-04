from typing import Tuple
import torch
from torch import nn
import torch.utils.data
import torchvision

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
            nn.Linear(latent_size, 64 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(-1, (64, 4, 4)),
            *generator_block(64, 32),
            *generator_block(32, 16, conv_kwargs=CONV_KWARGS_X4),
            *generator_block(16, 8, conv_kwargs=CONV_KWARGS_X4),
            *generator_block(
                8, 1, conv_kwargs=CONV_KWARGS_X4, add_norm_and_activation=False
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, latent_size, num_classes):
        super().__init__()

        assert latent_size % 16 == 0
        self.latent_size = latent_size
        self.num_classes = num_classes

        self.main = nn.Sequential(
            *discriminator_block(1, 8, conv_kwargs=CONV_KWARGS_X4),
            *discriminator_block(8, 16, conv_kwargs=CONV_KWARGS_X4),
            *discriminator_block(16, 32, conv_kwargs=CONV_KWARGS_X4),
            *discriminator_block(32, 64),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, num_classes),
        )

    def forward(self, x):
        return self.main(x)


class GAN(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        latent_size: int = 256,
        lr: float = 3e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_size)
        self.discriminator = Discriminator(latent_size, num_classes + 1)
        self.fixed_noise = torch.randn(36, latent_size)
        self.criterion = nn.CrossEntropyLoss()

    @property
    def device(self):
        return next(self.generator.parameters()).device

    def forward(self, noise):
        return self.generator(noise)

    def generate_images(self, batch_size: int):
        noise = torch.randn(batch_size, self.hparams.latent_size)  # type: ignore
        return self(noise.to(self.device))

    def generator_loss(self, imgs: torch.Tensor, labels: torch.Tensor, stage: str):
        fake_imgs = self.generate_images(len(imgs))
        preds = self.discriminator(fake_imgs)
        preds = torch.vstack((preds[:, :-1].sum(-1), preds[:, -1])).T
        loss = self.criterion(preds, torch.zeros_like(labels))

        self.log(f"loss/generator/{stage}", loss, on_step=False, on_epoch=True)

        return loss

    def discriminator_loss(self, imgs: torch.Tensor, labels: torch.Tensor, stage: str):
        # Compute discriminator loss on real data
        preds = self.discriminator(imgs)
        loss_real = self.criterion(preds, labels)
        acc_real = (preds.argmax(-1) == labels).float().mean()

        # Compute discriminator loss on fake data
        fake_imgs = self.generate_images(len(imgs))
        fake_labels = torch.full_like(labels, self.hparams.num_classes)  # type: ignore
        preds = self.discriminator(fake_imgs.detach())
        loss_fake = self.criterion(preds, fake_labels)
        acc_fake = (preds.argmax(-1) == fake_labels).float().mean()

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
        if batch_idx % 100 == 0:
            with torch.no_grad():
                imgs = self(self.fixed_noise.to(self.device))
            grid_size = int(len(self.fixed_noise) ** 0.5)
            grid = torchvision.utils.make_grid(imgs, nrow=grid_size, pad_value=1)
            self.logger.experiment.add_image(
                "fake_images_batch", grid, self.current_epoch * 1000 + batch_idx
            )

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

    def on_validation_epoch_end(self) -> None:
        # Log images to tensorboard
        with torch.no_grad():
            imgs = self(self.fixed_noise.to(self.device))
        grid_size = int(len(self.fixed_noise) ** 0.5)
        grid = torchvision.utils.make_grid(imgs, nrow=grid_size, pad_value=1)
        self.logger.experiment.add_image("fake_images", grid, self.current_epoch)
