import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, size=1):
        super(UnFlatten, self).__init__()
        self.size = size
    def forward(self, x):
        return x.view(x.size(0), -1, self.size, self.size)

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

# class Decoder(nn.Module):
#     def __init__(self, I=32, H=400, latent_size=64):
#         super(Decoder, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(latent_size, H),
#             nn.ReLU(inplace=True),
#             nn.Linear(H, I*I),
#             nn.Sigmoid(),
#             UnFlatten(size=I),
#         )
#     def forward(self, x):
#         x = self.model(x)
#         return x

class NAE(nn.Module):
    def __init__(self, I=32, H=400, latent_size=64, num_classes=10):
        super(NAE, self).__init__()

        self.Encoder = Encoder(I=I, H=H, latent_size=latent_size)
        self.Classifier = Classifier(latent_size=latent_size, num_classes=num_classes)

    def forward(self, x, test=False):
        z = self.Encoder(x)
        c = self.Classifier(z)
        return z, c

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




# class Encoder(nn.Module):
#     def __init__(self, chIn=1, feature=64):
#         super(Encoder, self).__init__()

#         self.model = nn.Sequential(
#             nn.Conv2d(chIn, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),

#             nn.AvgPool2d(kernel_size=4, stride=4),
#             Flatten(),
#             nn.Linear(256, feature),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature, feature),
#         )

#     def forward(self, x):
#         x = self.model(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self, chIn=1, feature=64):
#         super(Decoder, self).__init__()

#         self.model = nn.Sequential(
#             nn.Linear(feature, feature),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature, 256),
#             UnFlatten(size=2)   # 2*2*64

#             nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1) # 4*4*64
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1) # 8*8*32
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1) # 16*16*16
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(16, chIn, 3, stride=2, padding=1, output_padding=1) # 32*32*1
#             nn.Tanh(),
#         )
        
#     def forward(self, x):
#         x = self.model(x)
#         return x