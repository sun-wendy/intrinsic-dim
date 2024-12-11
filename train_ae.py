from typing import *
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

device = torch.device('cuda:0')
assert torch.cuda.is_available()

from dataset import Dataset
dataset = Dataset('train')

from models import Encoder, Decoder

def train_autoencoder(latent_dim: int):
    r'''
    Train encoder and decoder networks with `latent_dim` latent dimensions according
    to the autoencoder objective (i.e., MSE reconstruction).

    Returns the trained encoder and decoder.
    '''
    enc = Encoder(latent_dim).to(device)
    dec = Decoder(latent_dim).to(device)

    optim = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-4)

    dataset = Dataset('train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True)
    num_epochs = 30

    for epoch in tqdm(range(num_epochs), desc=f'{num_epochs} epochs total'):
        for batch, in dataloader:
            batch = batch.to(device)
            # batch: a batched image tensor of shape [B x 3 x 64 x 64]

            # FIXME
            loss = nn.MSELoss()(dec(enc(batch)), batch)

            optim.zero_grad()
            loss.backward()
            optim.step()
        # print(f'[Autoencoder] epoch {epoch: 4d}   loss = {loss.item():.4g}')

    return enc, dec


for dim in range(2, 65):
    ae_enc, ae_dec = train_autoencoder(dim)
    # save encoder and decoder for later use
    torch.save(ae_enc.state_dict(), f'models/ae_enc_dim{dim}.pth')
    torch.save(ae_dec.state_dict(), f'models/ae_dec_dim{dim}.pth')