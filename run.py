import os
from datetime import datetime

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from torchinfo import summary

from src import data
from src.models.ssgan import SSGAN
from src.models.cgan import CGAN

import pytorch_lightning as pl


def set_env():
    # The slurm job id is used to set the run name on tensorboard.
    os.environ["SLURM_JOB_ID"] = datetime.now().strftime("%Y%m%d%H%M%S")


def main():
    set_env()

    img_size = (128, 128)
    latent_size = 256

    datamodule = data.BreastCancerDataModule(batch_size=64, num_workers=0, preload=True)
    model = CGAN(latent_size=latent_size, lr=3e-4)

    summary(model.generator, input_size=(1, latent_size,))
    summary(model.discriminator, input_size=[(1, 1, *img_size), (1,)])

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2, monitor="accuracy/real/val", save_last=True
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=100,
        callbacks=[checkpoint_callback, TQDMProgressBar()],
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
