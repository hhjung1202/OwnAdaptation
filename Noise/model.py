import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class NoiseGradientLayerF(Function):

    @staticmethod
    def forward(ctx, x, gamma):
        ctx.gamma = gamma
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.gamma
        return output

class Discriminator(nn.Module):
    def __init__(self, chIn=1, clsN=10):
        super(Discriminator, self).__init__()

        def d_block(In, Out, stride=1):
            """Returns downsampling layers of each discriminator block"""

            layers = [  nn.Conv2d(In, Out, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(Out),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(Out, Out, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(Out),
                        nn.ReLU(inplace=True),]
            return layers

        self.init_model = nn.Sequential(
            nn.Conv2d(chIn, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.image_model = nn.Sequential(
            *d_block(16, 16),
            *d_block(16, 32, stride=2),           # 8
            *d_block(32, 32),
            *d_block(32, 64, stride=2),           # 8
            *d_block(64, 64),
            nn.AvgPool2d(kernel_size=8, stride=1),
        )

        self.condition_model = nn.Sequential(
            nn.Linear(clsN, 16),
            nn.Linear(16, 64),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        out = self.init_model(x)
        out = self.image_model(out).view(out.size(0), -1)

        y = self.condition_model(y)

        out = torch.cat([out,y], 1)
        out = self.fc(out)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
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

class Classifier(nn.Module):
    def __init__(self, chIn=1, clsN=10, resnet_layer=20):
        super(Classifier, self).__init__()

        self.init_model = nn.Sequential(
            nn.Conv2d(chIn, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        if resnet_layer is 14:
            self.n = 2
        elif resnet_layer is 20:
            self.n = 3
        elif resnet_layer is 32:
            self.n = 5
        elif resnet_layer is 44:
            self.n = 7
        elif resnet_layer is 56:
            self.n = 9
        elif resnet_layer is 110:
            self.n = 18
            

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_0', BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None))
        for i in range(1,self.n):
            self.layer1.add_module('layer1_%d' % (i), BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('layer2_0', BasicBlock(in_channels=16, out_channels=32, stride=2, downsample=True))
        for i in range(1,self.n):
            self.layer2.add_module('layer2_%d' % (i), BasicBlock(in_channels=32, out_channels=32, stride=1, downsample=None))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('layer3_0', BasicBlock(in_channels=32, out_channels=64, stride=2, downsample=True))
        for i in range(1,self.n):
            self.layer3.add_module('layer3_%d' % (i), BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None))

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, clsN)

    def forward(self, x, gamma):
        x = self.init_model(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        Noised_x = NoiseGradientLayerF.apply(x, gamma)
        return Noised_x


class Basic_Classifier(nn.Module):
    def __init__(self, chIn=1, clsN=10, resnet_layer=20):
        super(Basic_Classifier, self).__init__()

        self.init_model = nn.Sequential(
            nn.Conv2d(chIn, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        if resnet_layer is 14:
            self.n = 2
        elif resnet_layer is 20:
            self.n = 3
        elif resnet_layer is 32:
            self.n = 5
        elif resnet_layer is 44:
            self.n = 7
        elif resnet_layer is 56:
            self.n = 9
        elif resnet_layer is 110:
            self.n = 18
            
        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_0', BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None))
        for i in range(1,self.n):
            self.layer1.add_module('layer1_%d' % (i), BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('layer2_0', BasicBlock(in_channels=16, out_channels=32, stride=2, downsample=True))
        for i in range(1,self.n):
            self.layer2.add_module('layer2_%d' % (i), BasicBlock(in_channels=32, out_channels=32, stride=1, downsample=None))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('layer3_0', BasicBlock(in_channels=32, out_channels=64, stride=2, downsample=True))
        for i in range(1,self.n):
            self.layer3.add_module('layer3_%d' % (i), BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None))

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, clsN)

    def forward(self, x):
        x = self.init_model(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x