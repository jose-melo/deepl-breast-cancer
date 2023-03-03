from typing import Optional, Tuple

import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 1,
        val_frac: float = 0.2,
        img_size: Tuple[int, int] = (28, 28),
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.img_size = img_size
        self.random_state = random_state

    def _make_mnist(self, **kwargs):
        return MNIST(
            "data",
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(self.img_size)]
            ),
            **kwargs
        )

    def prepare_data(self):
        self.mnist_trainval = self._make_mnist(train=True, download=True)
        self.mnist_test = self._make_mnist(train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_trainval = self._make_mnist(train=True)
            generator = torch.Generator()
            if self.random_state is not None:
                generator = generator.manual_seed(self.random_state)
            self.mnist_train, self.mnist_val = random_split(
                mnist_trainval, [1 - self.val_frac, self.val_frac],
                generator=generator,
            )

        if stage == "test" or stage is None:
            self.mnist_test = self._make_mnist(train=False)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
