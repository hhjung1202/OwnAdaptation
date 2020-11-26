import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor

class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=4):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if isinstance(x, Tensor):
            input_tensor = x
        else:
            input_tensor = torch.cat(x,1)
        batch_size, num_channels, H, W = input_tensor.size()
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


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
        if self.count is 1:
            out = x
            x = [x]
        else:
            out = torch.cat(x,1)

        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return x + [out]

class _Basic(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, count):
        super(_Basic, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)
        self.count = count

    def forward(self, x):
        if self.count is 1:
            out = x
            x = [x]
        else:
            out = torch.cat(x,1)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv1(out)
        
        return x + [out]



class _Transition(nn.Sequential):
    def __init__(self, num_input_features, count=1):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_input_features // 2,
                        kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.se = ChannelSELayer(num_input_features // 2)
        
    def forward(self, x):
        if isinstance(x, Tensor):
            out = x
        else:
            out = torch.cat(x,1)
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        out = self.se(out)
        return out

class DenseNet_SE(nn.Module):

    def __init__(self, growth_rate=12,
                 num_init_features=24, num_classes=10, is_bottleneck=True, layer=22):
        super(DenseNet_SE, self).__init__()

        if layer is 28:
            block_config=[4,4,4]
        elif layer is 40:
            block_config=[6,6,6]
        elif layer is 52:
            block_config=[8,8,8]
        elif layer is 64:
            block_config=[10,10,10]

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
            self.features.add_module('se%d' % i, ChannelSELayer(num_features))

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