import torch
from models import ViTMNIST
import pprint
from torchinfo import summary

default_config = {
    "img_size": 128,
    "data_path": "data",
    "batch_size": 16,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu", 
    "n_channels": 1,
    "embedding_dim": 64, 
    "patch": 128,
    "dim_feed_forward": 128,
    "num_classes": 2,
    "num_layers": 6,
    "lr": 1e-4,
    "max_epochs": 50,
    "num_attention_heads": 4,
}

if __name__ == '__main__':

    print('Starting training: ')
    pprint.pprint(default_config)
    
    vit = ViTMNIST(default_config)
    summary(vit.model, (default_config["img_size"], default_config["n_channels"], default_config["img_size"], default_config["img_size"]))
    vit.train()


    

