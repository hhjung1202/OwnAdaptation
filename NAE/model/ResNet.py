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
    def __init__(self, block, num_blocks, memory, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.memory = memory
        self._forward = []

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self._make_layer(block, 64, num_blocks[0], stride=1)
        self._make_layer(block, 128, num_blocks[1], stride=2)
        self._make_layer(block, 256, num_blocks[2], stride=2)
        self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = Flatten()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            self._forward.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

    def forward(self, x):
        c = None
        out = F.relu(self.bn1(self.conv1(x)))
        
        for i, layer in enumerate(self._forward):
            out = layer(out)
            if i == self.memory:
                c = self.flatten(self.avgpool(out))

        out = self.avgpool(out)
        out = self.flatten(out)
        if c is None:
            c = out
        out = self.linear(out)
        return out, c


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])