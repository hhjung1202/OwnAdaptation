import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# class UnFlatten(nn.Module):
#     def __init__(self, size=1):
#         super(UnFlatten, self).__init__()
#         self.size = size
#     def forward(self, x):
#         return x.view(x.size(0), -1, self.size, self.size)

class Encoder(nn.Module):
    def __init__(self, I=32, H=400, latent_size=64):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(I*I, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, latent_size),
        )
    def forward(self, x):
        x = self.model(x)
        return x

class Classifier(nn.Module):
    def __init__(self, latent_size=64, num_classes=10):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(latent_size, num_classes),
        )

    def forward(self, z):
        z = self.model(z)
        return z

class NAE(nn.Module):
    def __init__(self, I=32, H=400, latent_size=64, num_classes=10):
        super(NAE, self).__init__()

        self.Encoder = Encoder(I=I, H=H, latent_size=latent_size)
        self.Classifier = Classifier(latent_size=latent_size, num_classes=num_classes)

    def forward(self, x, test=False):
        z = self.Encoder(x)
        out = self.Classifier(z)
        return out, z