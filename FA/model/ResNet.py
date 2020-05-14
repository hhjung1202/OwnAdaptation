import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, aug=1, num_classes=10, z=64):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.aug = aug

        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self._forward = ['init']

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_0', block(64, 64, 1))
        for i in range(1,num_blocks[0]):
            self.layer1.add_module('layer1_%d' % (i), block(64, 64))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('layer2_0', block(64, 128, 2))
        for i in range(1,num_blocks[1]):
            self.layer2.add_module('layer2_%d' % (i), block(128, 128))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('layer3_0', block(128, 256, 2))
        for i in range(1,num_blocks[2]):
            self.layer3.add_module('layer3_%d' % (i), block(256, 256))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('layer3_0', block(256, 512, 2))
        for i in range(1,num_blocks[3]):
            self.layer4.add_module('layer3_%d' % (i), block(512, 512))

        self.linear = nn.Linear(512, num_classes)

        # self.linear = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, num_classes),
        # )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = Flatten()

    def rotate(self, x):
        batch_size = x.size(0)
        num = batch_size // 4
        Rot_label = torch.zeros(batch_size)


        x0 = x[:num]
        x1 = x[num:2*num].transpose(2, 3)
        x2 = x[2*num:3*num].flip(2)
        x3 = x[3*num:].transpose(2, 3).flip(3)

        # Rot_label[:num] = 0
        # Rot_label[num:2*num] = 1
        # Rot_label[2*num:3*num] = 2
        # Rot_label[3*num:] = 3

        return torch.cat([x0,x1,x2,x3], dim=0)

    def forward(self, x):
        x = self.init(x)

        x = self.layer1(x)
        x = self.rotate(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.flatten(self.avgpool(x))
        x = self.fc(x)

        return x


def ResNet18(aug=1, num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], aug=aug, num_classes=num_classes)

def ResNet34(aug=1, num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], aug=aug, num_classes=num_classes)