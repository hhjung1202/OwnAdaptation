import torch
import torch.nn as nn
import torch.nn.functional as F
from .Feature_Adaptive import *
from .Contrastive_Loss import *

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ResNet(nn.Module):
    def __init__(self, blocks, num_blocks, num_classes):
        super(ResNet, self).__init__()

        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self._forward = []

        setattr(self, 'layer1_0', blocks(64, 64, 1));
        self._forward.append('layer1_0')
        for i in range(1, num_blocks[0]):
            setattr(self, 'layer1_%d' % (i), blocks(64, 64)); 
            self._forward.append('layer1_%d' % (i))

        setattr(self, 'layer2_0', blocks(64, 128, 2)); 
        self._forward.append('layer2_0')
        for i in range(1, num_blocks[1]):
            setattr(self, 'layer2_%d' % (i), blocks(128, 128)); 
            self._forward.append('layer2_%d' % (i))

        setattr(self, 'layer3_0', blocks(128, 256, 2)); 
        self._forward.append('layer3_0')
        for i in range(1, num_blocks[2]):
            setattr(self, 'layer3_%d' % (i), blocks(256, 256)); 
            self._forward.append('layer3_%d' % (i))

        setattr(self, 'layer4_0', blocks(256, 512, 2)); 
        self._forward.append('layer4_0')
        for i in range(1, num_blocks[3]):
            setattr(self, 'layer4_%d' % (i), blocks(512, 512)); 
            self._forward.append('layer4_%d' % (i))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = Flatten()
        self.linear = nn.Linear(512, num_classes)
        self.rot_lr = nn.Linear(512, 4)

    def forward(self, x):

        x = self.init(x)
        for i, name in enumerate(self._forward):
            layer = getattr(self, name)
            x = layer(x, None, None)
        x = self.flatten(self.avgpool(x))

        rot_cls = self.rot_lr(x)
        logits = self.linear(x)

        return rot_cls, logits

def ResNet18_rot(num_blocks=[2,2,2,2], num_classes=100):
    blocks = BasicBlock
    return ResNet(blocks, num_blocks, num_classes=num_classes)

def ResNet34_rot(num_blocks=[3,4,6,3], num_classes=100):
    blocks = BasicBlock
    return ResNet(blocks, num_blocks, num_classes=num_classes)
