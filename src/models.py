from sklearn.metrics import accuracy_score
import torch
from torch import nn
import  pytorch_lightning as pl

class EmbeddingLayer(nn.Module):

    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.config = config
        
        self.conv = nn.Conv2d( self.config["n_channels"], 
                               self.config["embedding_dim"], 
                               kernel_size=self.config["patch"],
                               stride=self.config["patch"]
                             )  
        
        self.cls = nn.Parameter(torch.zeros(1, 1, self.config["embedding_dim"]), requires_grad=True) 
        
        self.pos = nn.Parameter(torch.zeros(1, 
                                            (self.config["img_size"] // self.config["patch"]) ** 2 + 1,
                                                self.config["embedding_dim"]), 
                                            requires_grad=True)  
        
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        cls_token = torch.repeat_interleave(self.cls, x.shape[0], 0)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos
        return x

class AttentionLayer(nn.Module):
    
    def __init__(self, config):
        super(AttentionLayer, self).__init__()
        self.config = config
        self.config["head_embed_dim"] = self.config["embedding_dim"] // self.config["num_attention_heads"]

        self.proj_queries = nn.Linear(self.config["embedding_dim"], self.config["embedding_dim"], bias=True)
        self.proj_keys = nn.Linear(self.config["embedding_dim"], self.config["embedding_dim"], bias=True)
        self.proj_values = nn.Linear(self.config["embedding_dim"], self.config["embedding_dim"], bias=True)

        self.fc = nn.Linear(self.config["embedding_dim"], self.config["embedding_dim"])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        
        q = self.proj_queries(q)
        q = q.reshape(q.shape[0], q.shape[1], self.config["num_attention_heads"], self.config["head_embed_dim"])
        q = q.transpose(1, 2)
        q = q.reshape([-1, q.shape[2], q.shape[3]])

        k = self.proj_keys(k)
        k = k.reshape(k.shape[0], k.shape[1], self.config["num_attention_heads"], self.config["head_embed_dim"])
        k = k.transpose(1, 2)
        k = k.reshape([-1, k.shape[2], k.shape[3]])

        v = self.proj_values(v)
        v = v.reshape(v.shape[0], v.shape[1], self.config["num_attention_heads"], self.config["head_embed_dim"])
        v = v.transpose(1, 2)
        v = v.reshape([-1, v.shape[2], v.shape[3]])

        attention = q.bmm(k.transpose(1, 2))
        attention = attention / (self.config["head_embedding_dim"]**0.5)
        attention = torch.softmax(attention, dim=-1)

        x = attention.bmm(v)

        x = x.reshape([-1, self.config["num_attention_heads"], x.shape[1], x.shape[2]]) 
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        return x 

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
    
        self.attention = AttentionLayer(config)

        self.layer_norm1 = nn.LayerNorm(self.config["embedding_dim"])
        self.fc1 = nn.Linear(self.config["embedding_dim"], self.config["dim_feed_forward"])
        
        self.layer_norm2 = nn.LayerNorm(self.config["embedding_dim"])
        self.fc2 = nn.Linear( self.config["dim_feed_forward"], self.config["embedding_dim"])
    
        self.activation = nn.GELU()

    def forward(self, x):
        x_ = self.attention(x, x, x)
        x = x + x_
        x = self.layer_norm1(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = x + x_
        x = self.layer_norm2(x)

        return x

class ViT(pl.LightningModule):
    def __init__(self, config):
        super(ViT, self).__init__()
        self.save_hyperparameters()
        self.config = config
    
        self.embedding = EmbeddingLayer(config)

        self.encoding = nn.Sequential(
            *[
                Encoder(config)
                for _ in range(self.config["num_layers"])
            ],
            nn.LayerNorm(self.config["embedding_dim"]) 
            )
        
        self.fc1 = nn.Linear(self.config["embedding_dim"], self.config["embedding_dim"])
        self.fc2 = nn.Linear(self.config["embedding_dim"], self.config["num_classes"])
        
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoding(x)
        
        x = x[:, 0, :]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x 
    
            
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.config["lr"], weight_decay=1e-3)            

    def training_step(self, batch, batch_idx) -> None:
        loss = self.calculate_loss(batch, 'train')
        return loss
    
    def calculate_loss(self, batch, mode):
        imgs, labels = batch

        imgs, labels = imgs.to(self.config["device"]), labels.to(self.config["device"])

        predict = self.forward(imgs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(predict, labels)

        acc = accuracy_score(predict.argmax(dim=1).to("cpu"), labels.to('cpu')) * 100
        self.log("%s_loss" % mode, loss)
        self.log("%s_accuracy"%mode, acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.calculate_loss(batch, mode="val")
    
    def test_step(self, batch, batch_idx):
        self.calculate_loss(batch, mode="test")