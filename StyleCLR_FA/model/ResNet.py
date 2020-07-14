import torch
import torch.nn as nn
import torch.nn.functional as F
from .Feature_Adaptive import *

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Smoothing(nn.Module):
    def __init__(self, style_out):
        super(Smoothing, self).__init__()
        self.Gaussian = {   0: None,
                            1: GaussianSmoothing(64, 5, 1),
                            2: GaussianSmoothing(64, 5, 1),
                            3: GaussianSmoothing(128, 5, 1),
                            4: GaussianSmoothing(128, 5, 1),
                            5: GaussianSmoothing(256, 5, 1),
                            6: GaussianSmoothing(256, 5, 1),
                            7: GaussianSmoothing(512, 5, 1),
                            8: GaussianSmoothing(512, 5, 1),}[style_out]

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        output = self.Gaussian(x)
        return output


class ResNet(nn.Module):
    def __init__(self, blocks, num_blocks, style_out, num_classes=10, z=64):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.style_out = style_out

        index = 0

        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self._forward = []

        setattr(self, 'layer1_0', blocks[index](64, 64, 1)); index+=1
        self._forward.append('layer1_0')
        for i in range(1, num_blocks[0]):
            setattr(self, 'layer1_%d' % (i), blocks[index](64, 64)); index+=1
            self._forward.append('layer1_%d' % (i))

        setattr(self, 'layer2_0', blocks[index](64, 128, 2)); index+=1
        self._forward.append('layer2_0')
        for i in range(1, num_blocks[1]):
            setattr(self, 'layer2_%d' % (i), blocks[index](128, 128)); index+=1
            self._forward.append('layer2_%d' % (i))

        setattr(self, 'layer3_0', blocks[index](128, 256, 2)); index+=1
        self._forward.append('layer3_0')
        for i in range(1, num_blocks[2]):
            setattr(self, 'layer3_%d' % (i), blocks[index](256, 256)); index+=1
            self._forward.append('layer3_%d' % (i))

        setattr(self, 'layer4_0', blocks[index](256, 512, 2)); index+=1
        self._forward.append('layer4_0')
        for i in range(1, num_blocks[3]):
            setattr(self, 'layer4_%d' % (i), blocks[index](512, 512)); index+=1
            self._forward.append('layer4_%d' % (i))

        self.linear = nn.Linear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = Flatten()
        self.Smoothing = Smoothing(style_out)
        # self.L1Loss = torch.nn.L1Loss()
        self.MSELoss = nn.MSELoss()

    def forward(self, x, u_x):

        x_ = torch.cat([x, u_x], dim=0)
        x_ = self.init(x_)
        b = x_.size(0)
        x_ = x_.repeat(b+1, 1, 1, 1) # 1234, 1234, 1234, 1234

        for i, name in enumerate(self._forward):
            layer = getattr(self, name)
            x_ = layer(x_, b)

            if i+1 is self.style_out: # 2, 4, 6, 8
                style = x_[-b:]
                content_feat = x_[:-b]
        
        content_feat = x_[:-b]
        x = self.flatten(self.avgpool(content_feat))
        x = self.linear(x)

        return x, style_loss

def ResNet18(serial=[0,0,0,0,0,0,0,0], style_out=0, num_blocks=[2,2,2,2], num_classes=10):
    blocks = []
    for i in range(8):
        if serial[i] is 0:
            blocks.append(BasicBlock)
        elif serial[i] is 1:
            blocks.append(AdaptiveBlock)
        elif serial[i] is 2:
            blocks.append(PreBlock)
        elif serial[i] is 3:
            blocks.append(PostBlock)

    return ResNet(blocks, num_blocks, style_out=style_out, num_classes=num_classes)

def ResNet34(serial=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], style_out=0, num_blocks=[3,4,6,3], num_classes=10):
    blocks = []
    for i in range(8):
        if serial[i] is 0:
            blocks.append(BasicBlock)
        elif serial[i] is 1:
            blocks.append(AdaptiveBlock)
        elif serial[i] is 2:
            blocks.append(PreBlock)
        elif serial[i] is 3:
            blocks.append(PostBlock)

    return ResNet(blocks, num_blocks, style_out=style_out, num_classes=num_classes)