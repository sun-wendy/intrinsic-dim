from typing import *
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from torchvision import transforms
import torch.utils.data.dataloader as dataloader
from torchvision.datasets import FashionMNIST

device = torch.device('cuda:0')
assert torch.cuda.is_available()

SEED = 1
BATCH_SIZE = 128
NUM_EPOCHS = 20
LR = 1e-3
device = torch.device("cuda")
train = FashionMNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = dataloader.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

from models import VAE

def train_vae(z_dims: int):
    model = VAE(z_dims=z_dims).cuda()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for (X, _) in train_loader:
            X = X.cuda()
            optimizer.zero_grad()
            x_prime, mu, logvar = model(X)

            # FIXME: Calculate loss
            loss = ((x_prime - X) ** 2).sum() + -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
    
    return model

for z_dims in tqdm(range(2, 65)):
    vae = train_vae(z_dims)
    torch.save(vae.state_dict(), f'models/vae_dim{z_dims}.pth')
