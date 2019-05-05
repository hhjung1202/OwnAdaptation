import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator_Residual(nn.Module):
    def __init__(self, tgt_ch=3, src_ch=1, out_ch=3, num_classes=10, dim=32):
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

        self.tgt_encoder = nn.Sequential(
            nn.Conv2d(dim, 2*dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(2*dim, 4*dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(4*dim, 8*dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(8*dim, 8*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(inplace=True),
        )

        self.res_encoder = nn.Sequential(
            nn.Conv2d(2*dim, 2*dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(2*dim, 4*dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(4*dim, 8*dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(8*dim, 8*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(inplace=True),
        )

        self.tgt_decoder = nn.Sequential(
            nn.Conv2d(8*dim, 8*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8*dim, 4*dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),            

            nn.ConvTranspose2d(4*dim, 2*dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(2*dim, dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(dim, out_ch, 4, stride=2, padding=1),
            nn.Tanh(),
        )

        
        # self.src_decoder = nn.Sequential(
        #     nn.Conv2d(4*dim, 4*dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(4*dim),
        #     nn.ReLU(inplace=True),

        #     nn.ConvTranspose2d(4*dim, 2*dim, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(2*dim),
        #     nn.ReLU(inplace=True),
        # )

        num_fc = 8 * dim * 2**2
        self.fc = nn.Sequential(
            nn.Linear(num_fc, num_fc),
            nn.ReLU(inplace=True),
            nn.Linear(num_fc, num_classes),
        )

    def forward(self, src, tgt):
        src = self.src_init(src)
        tgt = self.tgt_init(tgt)

        res = torch.cat([src, tgt], 1)
        x = self.tgt_encoder(tgt)
        res = self.res_encoder(res)
        x = self.tgt_decoder(x + res)
        
        c = self.fc(res.view(res.size(0), -1))

        return x, c


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

