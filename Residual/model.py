import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator_Residual(nn.Module):
    def __init__(self, tgt_ch=3, out_tgt=3, out_src=1, y=10, dim=32):
        super(Generator_Residual, self).__init__()

        self.tgt_init = nn.Sequential(
            nn.Conv2d(tgt_ch, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.tgt_encoder = nn.Sequential(
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
            nn.Conv2d(dim + y, 2*dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(2*dim, 4*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(4*dim, 8*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(inplace=True),
        )

        self.tgt_decoder = nn.Sequential(
            nn.Conv2d(8*dim, 8*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(8*dim, 4*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(4*dim, 2*dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(inplace=True),            

            nn.ConvTranspose2d(2*dim, dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim, out_tgt, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.res_decoder = nn.Sequential(
            nn.Conv2d(8*dim, 8*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(8*dim, 4*dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(4*dim, 2*dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(inplace=True),            

            nn.ConvTranspose2d(2*dim, dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim, out_src, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def conv_y_concat(self, x, y):
        y = y.view(-1, y.size(1), 1, 1)
        x = torch.cat([x,y*torch.ones(x.size(0), y.size(1), x.size(2), x.size(3)).cuda()], 1)
        return x

    def forward(self, tgt, y):
        tgt = self.tgt_init(tgt)
        res = self.conv_y_concat(tgt, y) # 추가 실험을 통해서 여러 개로 두도록 학습시켜보자, dim 여러개로 학습을 해보자

        x = self.tgt_encoder(tgt)
        res = self.res_encoder(res)

        x = self.tgt_decoder(x + res)
        res = self.res_decoder(res)

        return x, res


class Discriminator(nn.Module):
    def __init__(self, input_ch=3, dim=32):
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