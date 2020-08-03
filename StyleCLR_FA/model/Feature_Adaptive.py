import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers

class Adain(nn.Module):
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        print(size)
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, feature, style_label, b):

        _, c, w, h = feature.size()
        print('1', feature.size())
        style = feature[-b:]
        print('1', style[0])
        print(style_label)
        style_feat = style[style_label].squeeze(0)
        print('1', style_feat[0])
        content_feat = feature[:-b]
        print('1', content_feat.size())
        
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean) / content_std
        adaptive_feat = normalized_feat * style_std + style_mean

        del(style_mean, style_std, content_mean, content_std)

        return torch.cat([adaptive_feat, style], dim=0)


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

    def forward(self, x, _, __):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AdaptiveBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(AdaptiveBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.adain = Adain()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, style_label, b):
        out = self.conv1(x)
        out = self.adain(out, style_label, b)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.adain(out, style_label, b)
        out += self.shortcut(x)
        out = F.relu(out)
        return out 

class PreBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.adain = Adain()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, style_label, b):
        out = self.conv1(x)
        out = self.adain(out, style_label, b)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out 

class PostBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PostBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.adain = Adain()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, style_label, b):
        out = self.conv1(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.adain(out, style_label, b)
        out += self.shortcut(x)
        out = F.relu(out)
        return out 
