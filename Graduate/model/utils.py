try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class _Gate(nn.Sequential):
    phase = True
    def __init__(self, channels, reduction, num_route):
        super(_Gate, self).__init__()
        self.num_route = num_route
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2 * channels, reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)    
        self.fc2 = nn.Linear(reduction, num_route, bias=False)
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