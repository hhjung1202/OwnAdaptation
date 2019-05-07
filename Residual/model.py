import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator_Residual(nn.Module):
    def __init__(self, tgt_ch=3, src_ch=1, out_ch=3, dim=32):
        super(Generator_Residual, self).__init__()

        # Initial convolution block
        self.src_init = nn.Sequential(
            nn.Conv2d(src_ch, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.tgt_init = nn.Sequential(
            nn.Conv2d(tgt_ch, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.src_encoder = nn.Sequential(
            nn.Conv2d(dim, 2*dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(2*dim, 4*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(4*dim, 8*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(inplace=True),

        )

        self.res_encoder = nn.Sequential(
            nn.Conv2d(2*dim, 2*dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(2*dim, 4*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(4*dim, 8*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(inplace=True),
        )

        self.src_decoder = nn.Sequential(
            nn.Conv2d(8*dim, 8*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(8*dim, 4*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(4*dim, 2*dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(inplace=True),            

            nn.ConvTranspose2d(2*dim, out_ch, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, src, tgt):
        src = self.src_init(src)
        tgt = self.tgt_init(tgt)
        x = self.src_encoder(src)
        residual = self.res_encoder(torch.cat([src, tgt], 1))
        x = x + residual
        x = self.src_decoder(x)
        return x

class Generator_Restore(nn.Module):
    def __init__(self, input_ch=3, out_ch=1, dim=32):
        super(Generator_Restore, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_ch, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.generator = nn.Sequential(
            nn.Conv2d(dim, 2*dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(2*dim, 4*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(4*dim, 4*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(4*dim, 2*dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(inplace=True),
        )

        self.out_deconv = nn.Sequential(
            nn.ConvTranspose2d(2*dim, out_ch, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, tgt):
        x = self.encoder(tgt)
        x = self.generator(x)
        x = self.out_deconv(x)
        return x

# 여기서 추후 작업 요망
class Discriminator(nn.Module):
    def __init__(self, input_ch=3, dim=24):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_ch, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(dim, 2*dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        fc_size = 2 * dim * 4**2
        self.fc1 = nn.Linear(fc_size, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size//4)
        self.fc3 = nn.Linear(fc_size//4, 1)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_ch=1, num_classes=10, dim=32):
        super(Classifier, self).__init__()

        def block(in_ch, out_ch, stride=1):
            layers = [  
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),  ]
            return layers

        self.model = nn.Sequential(
            *block(input_ch, dim, stride=1), 
            *block(dim, 2*dim, stride=2), 
            *block(2*dim, 4*dim, stride=2), 
            *block(4*dim, 8*dim, stride=2), 
        )

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(8*dim, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

