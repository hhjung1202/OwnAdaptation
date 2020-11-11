import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor


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

        return x + [out]

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
        
        return x + [out]

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, num_layers):
        super(_DenseLayer, self).__init__()
        self.norm = []
        self.layer = []
        self.gate = [nn.Sequential(),]
        self.num_layers = num_layers
        for i in range(num_layers):
            j = i//2 + 1
            self.norm.append(nn.BatchNorm2d(num_input_features + growth_rate * i))
            self.layer.append(_Basic(num_input_features + growth_rate * j, growth_rate))
            if i != num_layers-1:
                self.gate.append(_Gate_selection())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = [x]
        for i in range(self.num_layers):
            out = self.norm[i](torch.cat(x, 1))
            out = self.gate[i](out)
            out = self.layer[i](out)
        out = torch.cat(x,1)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv1(out)
        
        return x + [out]



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
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNet(nn.Module):

    def __init__(self, growth_rate=12,
                 num_init_features=24, num_classes=10, is_bottleneck=True, layer=22):
        super(DenseNet, self).__init__()

        if layer is 22:
            block_config=[3,3,3]
        elif layer is 28:
            block_config=[4,4,4]
        elif layer is 34:
            block_config=[5,5,5]
        elif layer is 40:
            block_config=[6,6,6]

        if is_bottleneck is not True:
            block_config = [2*x for x in block_config]

        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
        num_features = num_init_features

        for i in range(len(block_config)):
            
            layers = nn.Sequential()

            for j in range(block_config[i]):

                if is_bottleneck is True:
                    layer = _Bottleneck(num_features + j * growth_rate, growth_rate, count=j+1)
                    layers.add_module('layer%d_%d' % (i + 1, j + 1), layer)
                else:
                    layer = _Basic(num_features + j * growth_rate, growth_rate, count=j+1)
                    layers.add_module('layer%d_%d' % (i + 1, j + 1), layer)

            self.features.add_module('layer%d' % (i + 1), layers)
            num_features = num_features + block_config[i] * growth_rate

            if i != len(block_config) - 1:
                self.features.add_module('transition%d' % (i + 1), _Transition(num_features, count=block_config[i]+1))
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