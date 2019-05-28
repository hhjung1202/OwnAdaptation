import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, img_dim=3, latent_size=100, num_class=10, dim=128):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        self.encoder = Encoder(in_dim=img_dim, latent_size=latent_size, dim=dim)
        self.decoder = Decoder(out_dim=img_dim, latent_size=latent_size, dim=dim)
        self.inform = Inform(latent_size=latent_size, num_class=num_class)

    def forward(self, x, z=None):

        if z is not None:
            cls_output = self.inform(z)
            recon_x = self.decoder(z)
            return recon_x, cls_output

        batch_size = x.size(0)

        means, log_var = self.encoder(x)

        std = torch.exp(0.5 * log_var).cuda()
        eps = torch.randn([batch_size, self.latent_size]).cuda()

        z = eps * std + means

        cls_output = self.inform(z)
        recon_x = self.decoder(z)

        return recon_x, means, log_var, z, cls_output

class Encoder(nn.Module):
    def __init__(self, in_dim=3, latent_size=100, dim=128):
        super(Encoder, self).__init__()

        def block(In, Out, kernel_size=2, stride=2, padding=0):
            return nn.Sequential(
                nn.Conv2d(In, Out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(Out),
                nn.ReLU(inplace=True),
            )

        self.block1 = block(In=in_dim, Out=dim)
        self.block2 = block(In=dim, Out=2*dim)
        self.block3 = block(In=2*dim, Out=4*dim)

        self.linear_means = nn.Linear(4*dim * 4**2, latent_size)
        self.linear_log_var = nn.Linear(4*dim * 4**2, latent_size)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = x.view(x.size(0), -1)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars



class Decoder(nn.Module):
    def __init__(self, out_dim=3, latent_size=100, dim=128):
        super(Decoder, self).__init__()

        self.init = nn.Sequential(
            nn.Linear(latent_size, 4*dim * 4**2),
            nn.ReLU(inplace=True),
        )

        def block(In, Out, kernel_size=2, stride=2, padding=0):
            return nn.Sequential(
                nn.ConvTranspose2d(In, Out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(Out),
                nn.ReLU(inplace=True),            
            )

        self.block1 = block(In=4*dim, Out=4*dim)
        self.block2 = block(In=4*dim, Out=2*dim)
        self.block3 = block(In=2*dim, Out=dim)

        self.last = nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x = self.init(x)
        x = x.view(x.size(0), -1, 4, 4)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.last(x)

        return x

class Inform(nn.Module):
    def __init__(self, latent_size=100, num_class=10):
        super(Inform, self).__init__()

        self.fc1 = nn.Linear(latent_size, latent_size)
        self.fc2 = nn.Linear(latent_size, num_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
