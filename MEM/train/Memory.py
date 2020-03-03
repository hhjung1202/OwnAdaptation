import torch
import torch.distributions.normal as normal
import torch.nn.functional as F

class Memory(object):
    def __init__(self, args, Anchor=False):
        if Anchor:
            self.N = args.Anchor
        else:
            self.N = args.maxN # size of ALL Buffer
        self.index = 0
        self.Refine_N = int(args.Refine * args.maxN)
        # self.z = torch.zeros([self.N, args.z], device="cuda", dtype=torch.float32)
        self.z = torch.randn([self.N, args.z], device="cuda", dtype=torch.float32)
        self.z = F.normalize(self.z, p=2, dim=1)

    def Calc_Memory(self):
        self.mean = self.z.mean(dim=0)
        self.sigma = self.z.var(dim=0).sqrt()

    def Insert_memory(self, z): # Actual Function
        if self.index >= self.N:
            self.index = 0
        self.z[self.index] = z.data
        del(z)
        self.index = self.index + 1

class MemorySet(object):
    def __init__(self, args):
        self.clsN = args.clsN
        self.args = args
        self.Set = []
        self.AnchorSet = []
        for i in range(self.clsN):
            self.Set.append(Memory(args=args))
            self.AnchorSet.append(Memory(args=args, Anchor=True))

        self.mean_Set = torch.zeros((self.clsN, self.args.z), device="cuda", dtype=torch.float32)
        self.sigma_Set = torch.zeros((self.clsN, self.args.z), device="cuda", dtype=torch.float32)
        self.Anchor_mean_Set = torch.zeros((self.clsN, self.args.z), device="cuda", dtype=torch.float32)
        self.Anchor_sigma_Set = torch.zeros((self.clsN, self.args.z), device="cuda", dtype=torch.float32)

    def Batch_Insert(self, z, y, p):
        for i in range(z.size(0)):
            Pred_label = y[i]
            data = z[i]
            if p[i] > self.args.ramda:
                self.Set[Pred_label].Insert_memory(data)

        for i in range(self.clsN):
            self.Set[i].Calc_Memory()
            self.mean_Set[i] = self.Set[i].mean.detach()
            self.sigma_Set[i] = self.Set[i].sigma.detach()

    # def get_Gaussian_Percentage(self, Zn):
    #     # Scale.size = (Batch_size)
    #     P = torch.mean(self.Normal_Gaussian.cdf(Zn), dim=1) # 1-(P-0.5)*2 = 2-2P
    #     return 2-2*P

    def Calc_Pseudolabel(self, z, eps=1e-5):
        result = torch.zeros((z.size(0), self.clsN), device="cuda", dtype=torch.float32)

        for i in range(self.clsN):
            result[:, i] = (z - self.mean_Set[i]).pow(2).sum(dim=1)
        
        memory_soft_label = (result+eps).reciprocal().softmax(dim=1)
        memory_soft_label = memory_soft_label.pow(1/self.args.T) / memory_soft_label.pow(1/self.args.T).sum() # Sharpening
        return memory_soft_label.detach()

    def get_Regularizer(self, z, y, reduction='mean'):

        featureSet = z - self.mean_Set[y]
        Regularizer = featureSet.pow(2).sum().sqrt()

        if reduction == "mean":
            return Regularizer / z.size(0)
        elif reduction == "sum":
            return Regularizer

    def get_Regularizer_with_one(self, z, y, one, reduction='mean'):

        featureSet = z - self.mean_Set[y]
        Regularizer = (featureSet * one).pow(2).sum().sqrt()

        if reduction == "mean":
            return Regularizer / one.sum()
        elif reduction == "sum":
            return Regularizer * z.size(0) / one.sum()

