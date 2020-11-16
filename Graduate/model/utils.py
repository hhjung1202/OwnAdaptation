try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch
import torch.nn as nn
import itertools

class _Gate(nn.Sequential):
    phase = True
    def __init__(self, channels, reduction, num_route):
        super(_Gate, self).__init__()
        self.num_route = num_route
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2 * channels, reduction)
        self.relu = nn.ReLU(inplace=True)    
        self.fc2 = nn.Linear(reduction, num_route)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, res):
        
        x_ = self.avg_pool(x)
        res_ = self.avg_pool(res)
        out = torch.cat([x_,res_], 1)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, 2, 1, 1

        p = out[:,:1,:,:] # batch, 1, 1, 1
        q = out[:,1:,:,:] # batch, 1, 1, 1

        self.p = p.view(-1) / (p.view(-1) + q.view(-1))
        self.z = p / (p + q)
        return x * self.z + res * (1 - self.z)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class _Gate_selection(nn.Sequential):
    phase = 2
    def __init__(self, num_input_features, growth_rate, count, reduction=4):
        super(_Gate_selection, self).__init__()

        # self.growth_rate = growth_rate
        # self.init = num_init_features
        self.actual = (self.count-1) // 2 + 1
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

