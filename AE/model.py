import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    def __init__(self, img_D=784, H=400, latent_size=20, num_class=10):
        super(AE, self).__init__()

        self.latent_size = latent_size

        self.encoder = Encoder(D_in=img_D, H=H, latent_size=latent_size)
        self.decoder = Decoder(H=H, D_out=img_D, latent_size=latent_size)

    def forward(self, x, z=None):

        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, z

class Encoder(nn.Module):
    def __init__(self, D_in=784, H=400, latent_size=20):
        super(Encoder, self).__init__()

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, latent_size)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        z = self.linear2(x)
        return z

class Decoder(nn.Module):
    def __init__(self, H=400, D_out=784, latent_size=20):
        super(Decoder, self).__init__()
        
        self.linear1 = torch.nn.Linear(latent_size, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(x))