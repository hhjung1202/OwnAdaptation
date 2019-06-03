import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, img_D=784, H=100, D_out=100, latent_size=40, num_class=10):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        self.encoder = Encoder(D_in=img_D, H=H, D_out=D_out, latent_size=latent_size)
        self.decoder = Decoder(D_in=D_out, H=H, D_out=img_D, latent_size=latent_size)
        self.inform = Inform(latent_size=latent_size, num_class=num_class)

    def forward(self, x, z=None):

        print(x)
        print(x.size())

        x = x.view(x.size(0), -1)

        if z is not None:
            cls_output = self.inform(z)
            recover = self.decoder(z)
            return recover, cls_output

        batch_size = x.size(0)

        mean, log_sigma = self.encoder(x)

        sigma = torch.exp(log_sigma).cuda()
        eps = torch.randn([batch_size, self.latent_size]).cuda()

        z = eps * sigma + mean

        cls_output = self.inform(z)
        recover = self.decoder(z)

        return recover, mean, sigma, z, cls_output

class Encoder(nn.Module):
    def __init__(self, D_in=784, H=100, D_out=100, latent_size=40):
        super(Encoder, self).__init__()

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

        self.linear_mean = nn.Linear(D_out, latent_size)
        self.linear_log_sigma = nn.Linear(D_out, latent_size)

    def forward(self, x):

        x = self.linear1(x)
        x = self.linear2(x)

        mean = self.linear_mean(x)
        log_sigma = self.linear_log_sigma(x)

        return mean, log_sigma

class Decoder(nn.Module):
    def __init__(self, D_in=100, H=100, D_out=784, latent_size=40):
        super(Decoder, self).__init__()
        
        self.linear1 = torch.nn.Linear(latent_size, D_in)
        self.linear2 = torch.nn.Linear(D_in, H)
        self.linear3 = torch.nn.Linear(H, D_out)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.relu(self.linear3(x))

class Inform(nn.Module):
    def __init__(self, latent_size=40, num_class=10):
        super(Inform, self).__init__()

        self.fc1 = nn.Linear(latent_size, latent_size)
        self.fc2 = nn.Linear(latent_size, num_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
