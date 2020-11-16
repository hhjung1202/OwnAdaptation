import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
import itertools


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class _Gate_selection(nn.Sequential):
    phase = 2
    def __init__(self, num_input_features, growth_rate, count, reduction=4):
        super(_Gate_selection, self).__init__()

        # self.growth_rate = growth_rate
        # self.init = num_init_features
        self.actual = (count-1) // 2 + 1
        self.arr = [[i for i in range(num_input_features)]]
        s = num_input_features
        for j in range(count):
            self.arr += [[i for i in range(s, s + growth_rate)]]
            s+=growth_rate
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        channels = num_input_features + growth_rate * count
        self.fc1 = nn.Linear(channels, channels//reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels//reduction, count, bias=False)
        # self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()
        self.flat = Flatten()
        # self.split = [self.num_input_features] + [self.growth_rate] * self.actual


    def forward(self, x, x_norm):

        out = self.avg_pool(x_norm) # batch, channel 합친거, w, h
        out = self.flat(out)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        
        _, sort = out.sort()
        indices = sort[:,:self.actual] # batch, sort
        sliced_x = []
        for i in range(out.size(0)):
            select = [self.arr[0]]
            select += [self.arr[j+1] for j in indices[i]]
            select = list(itertools.chain.from_iterable(select))
            sliced_x += [x[i,select].unsqueeze(0)]

        sliced_x = torch.cat(sliced_x, 0)
        return sliced_x


class _Bottleneck(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, count=1):
        super(_Bottleneck, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, 4 * growth_rate,
                        kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)
        self.count = count

    def forward(self, x):
        if isinstance(x, Tensor):
            x = [x]
        out = torch.cat(x,1)

        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out

class _Basic(nn.Sequential):
    def __init__(self, num_input_features, growth_rate):
        super(_Basic, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)
        self.count = count

    def forward(self, x):
        if isinstance(x, Tensor):
            x = [x]
        out = torch.cat(x,1)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv1(out)
        
        return out

class _DevideFeature(nn.Module):
    def __init__(self, num_input_features, growth_rate):
        super(_DevideFeature, self).__init__()
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate

    def forward(self, x):
        count = (x.size(1) - self.num_input_features) // self.growth_rate
        split = [self.num_input_features] + [self.growth_rate] * count
        return x.split(split, dim=1)

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, num_layers, Block):
        super(_DenseLayer, self).__init__()
        self.norm = []
        self.layer = []
        self.gate = []
        self.num_layers = num_layers
        self.init_block = Block(num_input_features, growth_rate)
        for i in range(1, num_layers):
            j = (i-1)//2 + 1
            self.layer.append(Block(num_input_features + growth_rate * j, growth_rate))
            self.norm.append(nn.BatchNorm2d(num_input_features + growth_rate * (i+1)))
            self.gate.append(_Gate_selection(num_input_features, growth_rate, i+1, reduction=4))
        self.relu = nn.ReLU(inplace=True)

        self._devide = _DevideFeature(num_input_features, growth_rate)

    def forward(self, x):
        out = self.init_block(x)
        x = [x] + [out]
        self.print_f("init", x)
        out = torch.cat(x,1)
        for i in range(self.num_layers-1):
            out = self.layer[i](out)
            x += [out]
            self.print_f("layer", x)
            x_cat = torch.cat(x,1)
            t = self.norm[i](x_cat)
            out = self.gate[i](x_cat, t)
            print("gate", out.size())        
        return x

    def print_f(self, strs, s):
        for i in s:
            print(strs, i.size())

class _Transition(nn.Sequential):
    def __init__(self, num_input_features):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)

        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_input_features // 2,
                        kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = torch.cat(x,1)
        print(self.norm)
        print(out.size(1))
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNet(nn.Module):

    def __init__(self, growth_rate=12,
                 num_init_features=24, num_classes=10, is_bottleneck=True, layer=22):
        super(DenseNet, self).__init__()

        if layer is 28:
            block_config=[4,4,4]
        elif layer is 40:
            block_config=[6,6,6]

        if is_bottleneck:
            Block = _Bottleneck
        else:
            Block = _Basic
            block_config = [2*x for x in block_config]

        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
        num_features = num_init_features
        
        for i in range(len(block_config)):
            self.features.add_module('layer%d' % (i + 1), _DenseLayer(num_features, growth_rate, block_config[i], Block))
            num_features = num_features + block_config[i] * growth_rate
            if i != len(block_config) - 1:
                self.features.add_module('transition%d' % (i + 1), _Transition(num_features))
                num_features = num_features // 2

        # Final batch norm
        self.norm = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        # Linear layer
        # Official init from torch repo.

    def forward(self, x):
        out = self.features(x)
        out = torch.cat(out,1)
        out = self.norm(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__=='__main__':
    x = torch.randn(4,3,32,32)
    model = DenseNet(growth_rate=12, num_init_features=24, num_classes=10, is_bottleneck=True, layer=40)
    y = model(x)
    print(y.size())