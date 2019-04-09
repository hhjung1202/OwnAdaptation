import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) #
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), #avgPooling?
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out + x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, res_blocks=6, dim=64):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = dim
        out_features = dim*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.BatchNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResBlock(in_features, in_features, stride=1)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.Conv2d(in_features, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



# 여기서 추후 작업 요망
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, dim=64):
        super(Discriminator, self).__init__()

        def d_block(in_filters, out_filters, stride=2):
            """Returns downsampling layers of each discriminator block"""

            layers = [  nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(out_filters),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_filters),
                        nn.LeakyReLU(0.2, inplace=True)  ]
            return layers

        self.model = nn.Sequential(
            *d_block(in_channels, dim, stride=1), 
            *d_block(dim, 2*dim),
            *d_block(2*dim, 4*dim),
            *d_block(4*dim, 8*dim),
            nn.Conv2d(8*dim, 1, 4, padding=0)
        )

    def forward(self, x):
        return self.model(x)


class Classifier(nn.Module):
    def __init__(self, x_dim, num_classes=10, dim=64):
        super(Classifier, self).__init__()

        model = [   nn.Conv2d(x_dim, dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True) ]

        in_dim = dim
        out_dim = 2*dim
        for _ in range(3):
            model += [ResBlock(in_channels=in_dim, out_channels=out_dim, stride=2, downsample=True)]
            in_dim = out_dim
            out_dim = 2*in_dim

        self.model = nn.Sequential(*model)
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

