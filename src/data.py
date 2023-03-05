from typing import Callable, Optional, Tuple
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import MNIST
import torchvision
from sklearn.model_selection import train_test_split
from tqdm import tqdm

VAL_SIZE = 0.2
TEST_SIZE = 0.2


class BreastCancerDataset(Dataset):
    def __init__(
        self,
        root: str = "data/train_images_post",
        label_file: str = "data/train.csv",
        preload: bool = False,
        rebalance_positive: Optional[float] = None,
        augment: bool = False,
        _image_size: Tuple[int, int] = (512, 512)
    ):
        self.root = root
        self.rebalance_positive = rebalance_positive
        self.labels = pd.read_csv(
            label_file, index_col=False, usecols=["image_id", "cancer"]
        )
        if preload:
            self._preloaded = []
            for idx in tqdm(range(len(self.labels)), desc="Preloading data"):
                self._preloaded.append(self._load_idx(idx, convert_to_float=False))
        else:
            self._preloaded = None

        self.augment = augment
        if augment:
            self.random_resized_crop = transforms.RandomResizedCrop(
                _image_size, scale=(0.8, 1), ratio=(0.8, 1.2)
            )

        trainval, self.test = train_test_split(
            np.arange(len(self)),
            test_size=TEST_SIZE,
            stratify=self.labels.cancer,
            shuffle=True,
            random_state=42,
        )
        self.train, self.val = train_test_split(
            trainval,
            test_size=(VAL_SIZE) / (1 - TEST_SIZE),
            stratify=self.labels.iloc[trainval].cancer,
            shuffle=True,
            random_state=42,
        )

        self.train_neg = np.array(
            [idx for idx in self.train if self.labels.iloc[idx, 1] == 0]
        )
        self.train_pos = np.array(
            [idx for idx in self.train if self.labels.iloc[idx, 1] == 1]
        )

    def split(self) -> Tuple[Subset, Subset, Subset]:
        return (
            Subset(self, indices=self.train),
            Subset(self, indices=self.val),
            Subset(self, indices=self.test),
        )

    def __len__(self):
        return len(self.labels)

    def _load_idx(
        self,
        idx: int,
        convert_to_float: bool = False,
    ):
        image_id = self.labels.iloc[idx, 0]
        image_file = os.path.join(self.root, "0", f"{image_id}.png")
        img = torchvision.io.read_image(image_file)
        if convert_to_float:
            padded_img = torch.zeros((1, 512, 512), dtype=torch.float32)
            img = img.to(torch.float32) / 255
        else:
            padded_img = torch.zeros((1, 512, 512), dtype=torch.uint8)
        padded_img[[0], : img.shape[1], : img.shape[2]] = img
        return padded_img

    def _getitem(self, idx: int):
        if self._preloaded is not None:
            padded_img = self._preloaded[idx]
            padded_img = padded_img.to(torch.float32) / 255
        else:
            padded_img = self._load_idx(idx, convert_to_float=True)

        return padded_img

    def __getitem__(self, idx: int):
        # Resampling in the training dataset
        if self.rebalance_positive is not None and idx in self.train:
            if torch.rand(1) < self.rebalance_positive:
                idx = self.train_pos[torch.randint(len(self.train_pos), (1,))]
            else:
                idx = self.train_neg[torch.randint(len(self.train_pos), (1,))]

        img = self._getitem(idx)

        if self.augment:
            img = self.random_resized_crop(img)
            std = torch.rand(1) * 0.1
            img += torch.randn_like(img) * std.to(img.device)

        cancer = self.labels.iloc[idx, 1]
        return img, cancer


class BreastCancerDataset128(BreastCancerDataset):
    def __init__(
        self,
        root: str = "data/train_images_post",
        label_file: str = "data/train.csv",
        preload: bool = False,
        rebalance_positive: Optional[float] = None,
        augment: bool = False,
    ):
        super().__init__(
            root,
            label_file,
            preload=False,
            rebalance_positive=rebalance_positive,
            augment=augment,
            _image_size=(128, 128),
        )
        if preload:
            self._preloaded = []
            for idx in tqdm(range(len(self.labels)), desc="Preloading data"):
                self._preloaded.append(self._load_idx(idx).to("cuda"))

    def _load_idx(
        self,
        idx: int,
    ):
        img = super()._load_idx(idx, convert_to_float=True)
        return transforms.functional.resize(img, (128, 128))

    def _getitem(self, idx: int):
        if self._preloaded is not None:
            padded_img = self._preloaded[idx]
        else:
            padded_img = self._load_idx(idx)

        return padded_img


class BreastCancerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "data/train_images_post",
        label_file: str = "data/train.csv",
        batch_size: int = 64,
        num_workers: int = 1,
        resize_to_128: bool = True,
        preload: bool = False,
        rebalance_positive: Optional[float] = None,
        augment: bool = False,
    ):
        super().__init__()
        self.root = root
        self.label_file = label_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_to_128 = resize_to_128
        self.preload = preload
        self.rebalance_positive = rebalance_positive
        self.augment = augment

    def setup(self, stage=None):
        if self.resize_to_128:
            dataset_cls = BreastCancerDataset128
        else:
            dataset_cls = BreastCancerDataset

        self.train, self.val, self.test = dataset_cls(
            self.root,
            self.label_file,
            preload=self.preload,
            rebalance_positive=self.rebalance_positive,
            augment=self.augment,
        ).split()

    def _dataloader(self, ds: Subset):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._dataloader(self.train)

    def val_dataloader(self):
        return self._dataloader(self.val)

    def test_dataloader(self):
        return self._dataloader(self.test)


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
            **kwargs,
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
                mnist_trainval,
                [1 - self.val_frac, self.val_frac],
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
