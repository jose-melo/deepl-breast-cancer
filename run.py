import os
from datetime import datetime

from pytorch_lightning.callbacks.progress import TQDMProgressBar

from src import data, models

import pytorch_lightning as pl


def set_env():
    # The slurm job id is used to set the run name on tensorboard.
    os.environ["SLURM_JOB_ID"] = datetime.now().strftime("%Y%m%d%H%M%S")

def main():
    set_env()

    img_size = (256, 256)
    mnist = data.MNISTDataModule(
        batch_size=64, num_workers=4, val_frac=0.2, img_size=img_size,
    )
    model = models.GAN(img_size, num_classes=10, latent_size=256, lr=3e-4)

    trainer = pl.Trainer(gpus=1, max_epochs=20, callbacks=[TQDMProgressBar()])
    trainer.fit(model, mnist)


if __name__ == "__main__":
    main()
