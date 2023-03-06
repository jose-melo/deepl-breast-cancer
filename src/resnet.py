from torchvision import models, transforms
from torch import nn
import torch
import pytorch_lightning as pl

class Resnet18(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = models.resnet18(pretrained=True)
        self.n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.n_features, 2)

        for params in self.model.parameters():
            params.requires_grad = False 

        for params in self.model.fc.parameters():
            params.requires_grad = True

    def forward(self, inputs: torch.Tensor):
        data_transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.Tensor(x).repeat((1, 3, 1, 1))),
                transforms.Resize(232),
                transforms.CenterCrop(224),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        inputs = data_transform(inputs)
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train")
    
    def validation_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="val")
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.fc.parameters(), self.config["lr"], weight_decay=1e-3)
    
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

        


