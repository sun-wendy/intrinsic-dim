import torch
import torch.nn as nn
import torch.nn.functional as F

# encoder architecture
class Encoder(nn.Module):
    def __init__(self, latent_dim, normalize: bool = False):
        r'''
        latent_dim (int): Dimension of latent space
        normalize (bool): Whether to restrict the output latent onto the unit hypersphere
        '''
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1) # 64x64 --> 32x32
        self.conv2 = nn.Conv2d(32, 32*2, 4, stride=2, padding=1) # 32x32 --> 16x16
        self.conv3 = nn.Conv2d(32*2, 32*4, 4, stride=2, padding=1) # 16x16 --> 8x8
        self.conv4 = nn.Conv2d(32*4, 32*8, 4, stride=2, padding=1) # 8x8 --> 4x4
        self.conv5 = nn.Conv2d(32*8, 32*16, 4, stride=2, padding=1) # 4x4 --> 2x2
        self.conv6 = nn.Conv2d(32*16, latent_dim, 4, stride=2, padding=1) # 2x2 --> 1x1
        self.fc = nn.Linear(latent_dim, latent_dim)

        self.nonlinearity = nn.ReLU()
        self.normalize = normalize

    def forward(self, x):
        x = self.nonlinearity(self.conv1(x))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        x = self.nonlinearity(self.conv4(x))
        x = self.nonlinearity(self.conv5(x))
        x = self.nonlinearity(self.conv6(x).flatten(1))
        x = self.fc(x)
        if self.normalize:
            x = F.normalize(x)
        return x

    def extra_repr(self):
        return f'normalize={self.normalize}'


# decoder architecture
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        r'''
        latent_dim (int): Dimension of latent space
        '''
        super(Decoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(latent_dim, 32*16, 4, stride=2, padding=1) # 1x1 --> 2x2
        self.conv2 = nn.ConvTranspose2d(32*16, 32*8, 4, stride=2, padding=1) # 2x2 --> 4x4
        self.conv3 = nn.ConvTranspose2d(32*8, 32*4, 4, stride=2, padding=1) # 4x4 --> 8x8
        self.conv4 = nn.ConvTranspose2d(32*4, 32*2, 4, stride=2, padding=1) # 8x8 --> 16x16
        self.conv5 = nn.ConvTranspose2d(32*2, 32, 4, stride=2, padding=1) # 16x16 --> 32x32
        self.conv6 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1) # 32x32 --> 64x64
        self.nonlinearity = nn.ReLU()

    def forward(self, z):
        z = z[..., None, None]  # make it convolution-friendly

        x = self.nonlinearity(self.conv1(z))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        x = self.nonlinearity(self.conv4(x))
        x = self.nonlinearity(self.conv5(x))
        return self.conv6(x)
    
class VAE(nn.Module):
    def __init__(self, z_dims=4, input_size = 784, num_hidden=128):
        super().__init__()
        self.z_dims = z_dims
        self.input_size = input_size

        # FIXME: Create two encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU()
        )

        # FIXME: Create the mean and logvar readout layers
        self.mu = nn.Linear(num_hidden, z_dims)
        self.logvar = nn.Linear(num_hidden, z_dims)

        # FIXME: Create the decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(z_dims, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # FIXME: Implement the VAE forward function
        batch_size = x.shape[0]
        enc = self.encoder(x.view(batch_size, -1))
        mu = self.mu(enc)
        logvar = self.logvar(enc)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        x_prime = self.decoder(z)
        return x_prime.view(x.shape), mu, logvar