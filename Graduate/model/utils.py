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

if __name__=='__main__':
    x = torch.randn(3,34,2,2)
    model = _Gate_selection(num_input_features=10, growth_rate=4, count=6, reduction=4)
    y = model(x, x)
    print(y.size())
