import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, img_D=784, H=400, latent_size=20, num_class=10):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        self.encoder = Encoder(D_in=img_D, H=H, latent_size=latent_size)
        self.decoder = Decoder(H=H, D_out=img_D, latent_size=latent_size)
        self.inform = Inform(latent_size=latent_size, num_class=num_class)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, x, z=None):

        if z is not None:
            cls = self.inform(z)
            x_hat = self.decoder(z)
            return x_hat, cls

        x = x.view(x.size(0), -1)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        cls = self.inform(z)
        x_hat = self.decoder(z)

        return x_hat, mu, log_var, z, cls

class Encoder(nn.Module):
    def __init__(self, D_in=784, H=400, latent_size=20):
        super(Encoder, self).__init__()

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear_mean = nn.Linear(H, latent_size)
        self.linear_log_var = nn.Linear(H, latent_size)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        mu = self.linear_mean(x)
        log_var = self.linear_log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, H=400, D_out=784, latent_size=20):
        super(Decoder, self).__init__()
        
        self.linear1 = torch.nn.Linear(latent_size, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(x))

class Inform(nn.Module):
    def __init__(self, latent_size=20, num_class=10):
        super(Inform, self).__init__()
        self.fc1 = nn.Linear(latent_size, latent_size)
        self.fc1 = nn.Linear(latent_size, num_class)
    def forward(self, x):
        x = self.fc1(x)
        return x