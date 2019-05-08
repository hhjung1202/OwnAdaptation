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

        def block(in_filters, out_filters, kernel_size=4, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=1),
                nn.BatchNorm2d(out_filters),
                nn.ReLU(inplace=True),
            )

        self.res_encoder1 = block(in_filters=dim+y, out_filters=2*dim, kernel_size=4, stride=2)
        self.res_encoder2 = block(in_filters=2*dim+y, out_filters=4*dim, kernel_size=3, stride=1)
        self.res_encoder3 = block(in_filters=4*dim+y, out_filters=8*dim, kernel_size=3, stride=1)

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

        res = self.res_encoder1(self.conv_y_concat(tgt, y))
        res = self.res_encoder2(self.conv_y_concat(res, y))
        res = self.res_encoder3(self.conv_y_concat(res, y))

        x = self.tgt_encoder(tgt)
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