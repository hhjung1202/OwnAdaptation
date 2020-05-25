import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Adain(nn.Module):
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, content_feat):
        size = content_feat.size()

        style_feat = content_feat[torch.randperm(size[0])]
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.adain = Adain()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, is_adain):
        out = self.conv1(x)
        if is_adain:
            out = F.relu(self.adain(out))
        else:
            out = F.relu(self.bn1(out))

        out = self.conv2(x)
        if is_adain:
            out = self.adain(out)
        else:
            out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out, is_adain       

        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        # out = F.relu(out)
        # return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, z=64):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self._forward = []

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

    def forward(self, x, is_adain):
        x = self.init(x)
        for i, name in enumerate(self._forward):
            layer = getattr(self, name)
            x = layer(x, is_adain)

        x = self.flatten(self.avgpool(x))
        x = self.linear(x)

        return x


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)