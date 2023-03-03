import torch
from torch import nn

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

        self.proj_queries = nn.Linear(self.config["embedding_dim"], self.config["embedding_dim"], bias=True)
        self.proj_keys = nn.Linear(self.config["embedding_dim"], self.config["embedding_dim"], bias=True)
        self.proj_values = nn.Linear(self.config["embedding_dim"], self.config["embedding_dim"], bias=True)

        self.fc = nn.Linear(self.config["embedding_dim"], self.config["embedding_dim"])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        
        q = self.proj_queries(q)
        k = self.proj_keys(k)
        v = self.proj_values(v)

        attention = q.bmm(k.transpose(1, 2))
        attention = attention / (self.config["head_embedding_dim"]**0.5)

        x = attention.bmm(v)
        
        x = self.fc(x)

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

class ViT(nn.Module):
    def __init__(self, config):
        super(ViT, self).__init__()
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
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoding(x)
        
        x = x[:, 0, :]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.output(x)
        return x 
    