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

        self.actual = (count+1) // 2
        LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        
        self.init = LongTensor([i for i in range(num_input_features)]).view(1, -1)
        s = num_input_features
        arr = []
        for j in range(count):
            arr += [[i for i in range(s, s + growth_rate)]]
            s+=growth_rate
        self.arr = LongTensor(arr)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        channels = num_input_features + growth_rate * count
        self.fc1 = nn.Linear(channels, channels//reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels//reduction, count)
        self.sigmoid = nn.Sigmoid()
        self.flat = Flatten()

    def forward(self, x, x_norm):
        b, _, w, h = x_norm.size()
        out = self.avg_pool(x_norm) # batch, channel 합친거, w, h
        out = self.flat(out)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        
        _, sort = out.sort()
        indices = sort[:,:self.actual] # batch, sort # shuffle
        indices = indices[:, torch.randperm(indices.size(1))]

        select = self.init.repeat(b,1)
        select = torch.cat([select, self.arr[indices].view(b,-1)], 1)
        select = select.view(select.size(0), -1, 1, 1).repeat(1,1,w,h)

        x = x.gather(1, select)
        return x

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

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, num_layers, Block):
        super(_DenseLayer, self).__init__()
        self.num_layers = num_layers
        self.init_block = Block(num_input_features, growth_rate)
        for i in range(1, num_layers):
            j = (i-1)//2 + 1
            setattr(self, 'layer{}'.format(i), Block(num_input_features + growth_rate * j, growth_rate))
            setattr(self, 'norm{}'.format(i), nn.BatchNorm2d(num_input_features + growth_rate * (i+1)))
            setattr(self, 'gate{}'.format(i), _Gate_selection(num_input_features, growth_rate, i+1, reduction=4))

    def forward(self, x):
        out = self.init_block(x)
        x = [x] + [out]
        out = torch.cat(x,1)
        for i in range(1, self.num_layers):
            out = getattr(self, 'layer{}'.format(i))(out)
            x += [out]
            x_cat = torch.cat(x,1)
            x_norm = getattr(self, 'norm{}'.format(i))(x_cat)
            out = getattr(self, 'gate{}'.format(i))(x_cat, x_norm)
        return x_cat

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, tr_features):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(tr_features)

        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(tr_features, num_input_features // 2,
                        kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # out = torch.cat(x,1)
        out = self.norm(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNet(nn.Module):

    def __init__(self, growth_rate=12,
                 num_init_features=24, num_classes=10, is_bottleneck=True, layer=28):
        super(DenseNet, self).__init__()

        if layer is 28:
            block_config=[4,4,4]
        elif layer is 40:
            block_config=[6,6,6]
        elif layer is 52:
            block_config=[8,8,8]
        elif layer is 64:
            block_config=[10,10,10]

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
            tr_features = num_features + block_config[i] * growth_rate
            num_features = num_features + block_config[i] * growth_rate // 2
            if i != len(block_config) - 1:
                self.features.add_module('transition%d' % (i + 1), _Transition(num_features, tr_features))
                num_features = num_features // 2

        # Final batch norm
        self.norm = nn.BatchNorm2d(tr_features)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(tr_features, num_classes)

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
        # out = torch.cat(out,1)
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