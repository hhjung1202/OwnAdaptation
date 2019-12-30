import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# class UnFlatten(nn.Module):
#     def __init__(self, size=1):
#         super(UnFlatten, self).__init__()
#         self.size = size
#     def forward(self, x):
#         return x.view(x.size(0), -1, self.size, self.size)

class Block_A(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(Block_A, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels) #
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        
    def forward(self, x):
        out = F.relu(self.bn1(x))
        h1 = self.conv1(out)
        h1 = self.conv2(F.relu(self.bn2(h1)))
        h2 = self.conv3(out)
        return h1+h2

class Block_B(nn.Module):

    def __init__(self, in_channels):
        super(Block_B, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels) #
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        return h + x


class PreActResNet(nn.Module):
    def __init__(self, num_classes=10, resnet_layer=32, memory=0):
        super(PreActResNet, self).__init__()

        filters = [32, 32, 64, 128]
        self.memory = memory
        self.conv = nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1, bias=False)

        if resnet_layer is 14:
            self.n = 2
        elif resnet_layer is 20:
            self.n = 3
        elif resnet_layer is 32:
            self.n = 5
            # memory 15, else
        elif resnet_layer is 44:
            self.n = 7
        elif resnet_layer is 56:
            self.n = 9
        elif resnet_layer is 110:
            self.n = 18

        self._forward = ["conv"]
            
        setattr(self, 'layer1_0', Block_A(in_channels=filters[0], out_channels=filters[1], stride=1))
        self._forward.append('layer1_0')
        for i in range(1,self.n):
            setattr(self, 'layer1_%d' % (i), Block_B(in_channels=filters[1]))
            self._forward.append('layer1_%d' % (i))

        setattr(self, 'layer2_0', Block_A(in_channels=filters[1], out_channels=filters[2], stride=2))
        self._forward.append('layer2_0')
        for i in range(1,self.n):
            setattr(self, 'layer2_%d' % (i), Block_B(in_channels=filters[2]))
            self._forward.append('layer2_%d' % (i))

        setattr(self, 'layer3_0', Block_A(in_channels=filters[2], out_channels=filters[3], stride=2))
        self._forward.append('layer3_0')
        for i in range(1,self.n):
            setattr(self, 'layer3_%d' % (i), Block_B(in_channels=filters[3]))
            self._forward.append('layer3_%d' % (i))

        self.bn = nn.BatchNorm2d(filters[3])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(filters[3], num_classes)

        self.flatten = Flatten()

    def forward(self, x):
        z = None
        for i, name in enumerate(self._forward):
            layer = getattr(self, name)
            x = layer(x)
            if i == self.memory:
                z = self.flatten(self.avgpool(x))

        x = F.relu(self.bn(x))
        x = self.avgpool(x)
        x = self.flatten(x)

        if z is None:
            z = x

        out = self.fc(x)
        return out, z