try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch
import torch.nn as nn

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
        self.count = count
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        channels = num_init_features + growth_rate * count
        self.fc1 = nn.Linear(channels, channels//reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels//reduction, self.count, bias=False)
        # self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()
        self.p = None
        self.flat = Flatten()

    def forward(self, x, t):

        out = torch.cat(t,1)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        

        _, indices = out.sort()
        select_indices = indices[:,:(self.count-1) // 2 + 1] # batch, indices

        batch x[select_indices][0] cat

        arr = []
        for i, k in enumerate(select_indices):
            arr.append([])
            for j in k:
                arr[i].append(x[j][i])
            arr[i] = torch.cat(arr[i], 1)


        return torch.cat(x,1)


class _Gate2(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, count):
        super(_Gate2, self).__init__()

        # self.growth_rate = growth_rate
        # self.init = num_init_features
        self.count = count
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduction, self.count, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()
        self.p = None

    def forward(self, x):

        out = torch.cat(x,1)
        out = self.avg_pool(out)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, count, 1, 1

        # out size is   # batch, count, 1, 1
        # x size is     # count * [batch, 16, 32, 32]

        self.p = list(torch.split(out, 1, dim=1)) # array of [batch, 1, 1, 1]
        p_sum = sum(self.p) # [batch, 1, 1, 1]

        for i in range(self.count):
            self.p[i] = self.p[i] / p_sum * self.count / 2
            x[i] = x[i] * self.p[i]

        return torch.cat(x,1)

class _Gate3(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, count):
        super(_Gate3, self).__init__()

        # self.growth_rate = growth_rate
        # self.init = num_init_features
        self.count = count
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduction, self.count, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()
        self.p = None

    def forward(self, x):

        out = torch.cat(x,1)
        out = self.avg_pool(out)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, count, 1, 1

        # out size is   # batch, count, 1, 1
        # x size is     # count * [batch, 16, 32, 32]

        self.p = list(torch.split(out, 1, dim=1)) # array of [batch, 1, 1, 1]
        p_sum = sum(self.p) # [batch, 1, 1, 1]

        for i in range(self.count):
            self.p[i] = self.p[i] / p_sum * 3 # normalize
            x[i] = x[i] * self.p[i]

        return torch.cat(x,1)