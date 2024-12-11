#%%
import skdim
import numpy as np
import torch 
import matplotlib.pyplot as plt
from models import Encoder, Decoder
from dataset import Dataset
from typing import Optional

device = torch.device('cuda')

train_dataset = Dataset('train')
val_dataset = Dataset('val')
_ = train_dataset.visualize_multiple(np.arange(10))
#%%
@torch.no_grad()
def get_features(dataset: Dataset, encoder: Encoder, firstk: Optional[int] = None):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, pin_memory=True)
    n = 0
    features = []
    for batches in dataloader:
        batch = batches[0]
        features.append(encoder(batch.to(device)))
        n += batch.shape[0]
        if firstk is not None and n >= firstk:
            break
    features = torch.cat(features)
    if firstk is not None:
        features = features[:firstk]
    return features

@torch.no_grad()
def get_loss(dataset: Dataset, encoder: Encoder, decoder):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, pin_memory=True)
    loss = 0
    total = 0
    for batches in dataloader:
        batch = batches[0]
        batch = batch.to(device)
        features = encoder(batch)
        recon = decoder(features)
        loss += torch.nn.functional.mse_loss(recon, batch, reduction='sum').item()        
        total += batch.shape[0]
    return loss / total
#%%
def get_intrinsic_dimension(act):
    corr_result = skdim.id.CorrInt().fit_transform(act)
    mle_result = skdim.id.MLE().fit_transform(act)
    twoNN_result = skdim.id.TwoNN().fit_transform(act)
    lpca_result = skdim.id.lPCA(alphaFO=0.225).fit_transform(act)

    return float(corr_result), float(mle_result), float(twoNN_result), float(lpca_result)

import sklearn
def get_actual_dimension(act):
    pca = sklearn.decomposition.PCA()
    pca.fit(act)
    evr = pca.explained_variance_ratio_
    return int(np.argmax(np.cumsum(evr) >= 0.95) + 1)
#%%
def analyze_dim(dim):
    # load model
    enc = Encoder(dim).to(device)
    dec = Decoder(dim).to(device)
    enc.load_state_dict(torch.load(f'models/ae_enc_dim{dim}.pth'))
    dec.load_state_dict(torch.load(f'models/ae_dec_dim{dim}.pth'))

    # compute train/val loss and accuracy
    train_loss = get_loss(train_dataset, enc, dec)
    val_loss = get_loss(val_dataset, enc, dec)

    features = get_features(train_dataset, enc, 10000).cpu().numpy()
    corr, mle, twoNN, lpca = get_intrinsic_dimension(features)
    pca = get_actual_dimension(features) 
    return {
        'dim': dim,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'CorrInt': corr,
        'MLE': mle,
        'TwoNN': twoNN,
        'lPCA': lpca,
        'PCA': pca
    }
#%%
# construct a dataframe of results
import pandas as pd
from tqdm import tqdm

entries = []
for dim in tqdm(range(2, 65)):
    entries.append(analyze_dim(dim))

df = pd.DataFrame(entries)
df.to_csv('idim_results.csv', index=False)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

def format_subplot(ax, grid_x=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid_x:
        ax.grid(linestyle='--', alpha=0.4)
    else:
        ax.grid(axis='y', linestyle='--', alpha=0.4)
#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=600)

# use sns.lineplot to plot the intrinsic dimensions on ax[0]
# one line for each method: CorrInf, MLE, TwoNN, lPCA, PCA
# first convert df to melt
melted = df.melt(id_vars='dim', value_vars=['CorrInt', 'MLE', 'TwoNN', 'lPCA', 'PCA'], var_name='method', value_name='value')
sns.lineplot(data=melted, x='dim', y='value', hue='method', ax=axes[0], alpha=0.8)
format_subplot(axes[0])

sns.lineplot(data=df, x='dim', y='val_loss', ax=axes[1], label='Validation loss', alpha=0.8)
format_subplot(axes[1])
# %%
