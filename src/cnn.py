import torch
from torch import nn
import pytorch_lightning as pl

class CNN(pl.LightningModule):
    def __init__(self, config) -> None:
        super(CNN, self).__init__()
        self.config = config

        self.conv1 = nn.Conv2d(self.config["n_channels"], self.config["c1"], self.config["kernel_size"], stride=1)
        self.conv2 = nn.Conv2d(self.config["c1"], self.config["c2"], self.config["kernel_size"], stride=2)
        self.conv3 = nn.Conv2d(self.config["c2"], self.config["c3"], self.config["kernel_size"], stride=2)
        self.conv4 = nn.Conv2d(self.config["c3"], self.config["c4"], self.config["kernel_size"], stride=2)

        self.batchnorm = nn.BatchNorm2d(self.config["c2"])
        self.size_fc = self.config["c4"]*self.config["size"]*self.config["size"]
        self.linear1 = nn.Linear(self.size_fc, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

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
        return x
    
    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train")
    
    def validation_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="val")
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.config["lr"], weight_decay=1e-3)
    
    def calculate_loss(self, batch, mode):
        imgs, labels = batch

        imgs, labels = imgs.to(self.config["device"]), labels.to(self.config["device"])

        predict = self.forward(imgs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(predict, labels)

        prediction = torch.softmax(predict, dim = 1).argmax(dim=1)
        m = {}
        m["tp"] = ((prediction == 1) & (labels == 1)).float().sum()
        m["fp"] = ((prediction == 1) & (labels == 0)).float().sum()
        m["tn"] = ((prediction == 0) & (labels == 0)).float().sum()
        m["fn"] = ((prediction == 0) & (labels == 1)).float().sum()

        m["precision"] = m["tp"] / (m["tp"] + m["fp"])
        m["recall"] = m["tp"] / (m["tp"] + m["fn"])
        m["f1"] = 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])

        self.log("%s_loss" % mode, loss)
        for metric, value in m.items():
            self.log_dict(
                {
                    "ext/" + metric + "_" + mode: float(value),
                    r"step": float(self.current_epoch),
                },
                on_step=False,
                on_epoch=True,
                reduce_fx=torch.sum
            )

        return loss

        


