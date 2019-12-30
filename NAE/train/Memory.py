import torch
import torch.distributions.normal as normal

class Memory(object):
    def __init__(self, args):
        self.N = args.maxN # size of ALL Buffer
        self.index = 0
        self.index2 = 0
        self.z = torch.zeros([self.N, args.z], device="cuda", dtype=torch.float32)
        self.vector = torch.zeros([self.N, args.z], device="cuda", dtype=torch.float32)

    def Calc_Vector(self, eps=1e-9): # After 1 Epoch, it will calculated
        mean_len = self.vector.mean(dim=0).pow(2).sum().sqrt() + eps
        len_mean = self.vector.pow(2).sum(dim=1).sqrt().mean()
        self.mean_v = self.vector.mean(dim=0) * len_mean / mean_len
        self.sigma_v = self.vector.var(dim=0).sqrt()
        self.len_v = len_mean

    def Calc_Memory(self): # After 1 Epoch, it will calculated
        self.mean = self.z.mean(dim=0)
        self.sigma = self.z.var(dim=0).sqrt()
        return self.mean

    def Insert_memory(self, z): # Actual Function
        if self.index >= self.N:
            self.index = 0
        self.z[self.index] = z.data
        del(z)
        self.index = self.index + 1

    def Insert_vector(self, vector): # Actual Function
        if self.index2 >= self.N:
            self.index2 = 0
        self.vector[self.index2] = vector.data
        del(vector)
        self.index2 = self.index2 + 1

class MemorySet(object):
    def __init__(self, args):
        self.Normal_Gaussian = normal.Normal(0,1) # mean 0, var 1
        self.clsN = args.clsN
        self.Set = []
        self.size_z = args.z
        for i in range(self.clsN):
            self.Set.append(Memory(args=args))

        self.mean_v_Set = torch.zeros((self.clsN, self.size_z), device="cuda", dtype=torch.float32)
        self.len_v_Set = torch.zeros((self.clsN), device="cuda", dtype=torch.float32)
        self.sigma_v_Set = torch.zeros((self.clsN, self.size_z), device="cuda", dtype=torch.float32)

    def Batch_Insert(self, z, y):
        for i in range(z.size(0)):
            Noise_label = y[i]
            data = z[i]
            self.Set[Noise_label].Insert_memory(data)

        self.Calc_Center()
        self.Batch_Vector_Insert(z, y)

        for i in range(self.clsN):
            self.Set[i].Calc_Vector()
            self.mean_v_Set[i] = self.Set[i].mean_v.detach()
            self.len_v_Set[i] = self.Set[i].len_v.detach()
            self.sigma_v_Set[i] = self.Set[i].sigma_v.detach()

    def Batch_Vector_Insert(self, z, y):
        vectorSet = z - self.T.detach()
        for i in range(vectorSet.size(0)):
            Noise_label = y[i]
            vector = vectorSet[i]
            self.Set[Noise_label].Insert_vector(vector)

    def Calc_Center(self):
        self.T = torch.zeros(self.size_z, device='cuda', dtype=torch.float32)
        for i in range(self.clsN):
            self.T += self.Set[i].Calc_Memory()
        self.T = (self.T / self.clsN)

# --------------------------

    def get_DotLoss_Noise(self, z, y, reduction='mean', reverse=False):
        # Y(1-cos)
        vectorSet = z - self.T.detach()
        if reverse:
            vectorSet = -vectorSet

        len_v = vectorSet.pow(2).sum(dim=1).sqrt()
        Dot = torch.sum(vectorSet * self.mean_v_Set[y], dim=1)
        loss = torch.sum(self.len_v_Set[y] - Dot / len_v)

        if reduction == "mean":
            return loss / z.size(0)
        elif reduction == "sum":
            return loss

    def get_DotLoss_BASE(self, z, y, reduction='mean', reverse=False):
        # XY(1-cos)
        vectorSet = z - self.T.detach()
        if reverse:
            vectorSet = -vectorSet

        len_v = vectorSet.pow(2).sum(dim=1).sqrt()
        Dot = torch.sum(vectorSet * self.mean_v_Set[y], dim=1)
        loss = torch.sum(len_v * self.len_v_Set[y] - Dot)

        if reduction == "mean":
            return loss / z.size(0)
        elif reduction == "sum":
            return loss

# --------------------------

    # def get_Gaussian_Percentage(self, Zn):
    #     # Scale.size = (Batch_size)
    #     P = torch.mean(self.Normal_Gaussian.cdf(Zn), dim=1) # 1-(P-0.5)*2 = 2-2P
    #     return 2-2*P

    def Calc_Pseudolabel(self, z, y):
        vectorSet = z - self.T.detach()
        cos = torch.nn.CosineSimilarity(dim=1)
        cos_result = torch.zeros((z.size(0), self.clsN), device="cuda", dtype=torch.float32)

        for i in range(self.clsN):
            mean_vector = self.mean_v_Set[i].unsqueeze(0).repeat(z.size(0), 1)
            cos_result[:, i] = cos(vectorSet, mean_vector)

        _, pseudo_label = cos_result.max(1)

        return pseudo_label.detach()

    def get_Regularizer(self, z):
        vectorSet = z - self.T.detach()
        len_v = vectorSet.pow(2).sum(dim=1).sqrt()
        s = torch.pow(torch.sum(len_v) / z.size(0), 2) # E(X)^2
        ss = torch.sum(len_v.pow(2)) / z.size(0)       # E(X^2)
        Regularizer = ss - s
        return Regularizer

    # def Test_Init(self):
    #     for i in range(self.clsN):
    #         self.Set[i].Calc_Vector()
    #         self.mean_v_Set[i] = self.Set[i].mean_v.detach()
    #         self.len_v_Set[i] = self.Set[i].len_v.detach()
    #         self.sigma_v_Set[i] = self.Set[i].sigma_v.detach()

    # def Calc_Test_Similarity(self, z, y):
    #     vectorSet = z - self.T.detach() # Test Image Save
    #     Sim_scale = torch.tensor(0, device='cuda', dtype=torch.float32)
    #     Sim_vector = torch.tensor(0, device='cuda', dtype=torch.float32)

    #     cos = torch.nn.CosineSimilarity(dim=1)
    #     Sim_scale = torch.sum(torch.abs((vectorSet - self.mean_v_Set[y]))/self.sigma_v_Set[y]) / z.size(0)
    #     Sim_vector = torch.sum(torch.abs(cos(vectorSet, self.mean_v_Set[y])))

    #     return Sim_scale, Sim_vector
