import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, memory=1, num_classes=10, z=64):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.memory = memory

        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self._forward = ['init']

        setattr(self, 'layer1_0', block(64, 64, 1))
        self._forward.append('layer1_0')
        for i in range(1, num_blocks[0]):
            setattr(self, 'layer1_%d' % (i), block(64, 64))
            self._forward.append('layer1_%d' % (i))

        setattr(self, 'layer2_0', block(64, 128, 2))
        self._forward.append('layer2_0')
        for i in range(1, num_blocks[1]):
            setattr(self, 'layer2_%d' % (i), block(128, 128))
            self._forward.append('layer2_%d' % (i))

        setattr(self, 'layer3_0', block(128, 256, 2))
        self._forward.append('layer3_0')
        for i in range(1, num_blocks[2]):
            setattr(self, 'layer3_%d' % (i), block(256, 256))
            self._forward.append('layer3_%d' % (i))

        setattr(self, 'layer4_0', block(256, 512, 2))
        self._forward.append('layer4_0')
        for i in range(1, num_blocks[3]):
            setattr(self, 'layer4_%d' % (i), block(512, 512))
            self._forward.append('layer4_%d' % (i))

        self.linear = nn.Linear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = Flatten()

    def forward(self, x):
        z = None
        for i, name in enumerate(self._forward):
            layer = getattr(self, name)
            x = layer(x)
            if i == self.memory:
                z = self.flatten(self.avgpool(x))

        x = self.avgpool(x)
        x = self.flatten(x)
        if z is None:
            z = x
        out = self.linear(x)
        return out, z


def ResNet18(memory=1, num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], memory=memory, num_classes=num_classes)

def ResNet34(memory=1, num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], memory=memory, num_classes=num_classes)