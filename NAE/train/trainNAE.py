import argparse
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import os
import torch.backends.cudnn as cudnn
import time
import utils
import dataset
import math
import torch.distributions.normal as normal

def to_var(x, dtype):
    return Variable(x.type(dtype))

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
        vectorSet = z - self.T
        for i in range(vectorSet.size(0)):
            Noise_label = y[i]
            vector = vectorSet[i]
            self.Set[Noise_label].Insert_vector(vector)

    def Calc_Center(self):
        self.T = torch.zeros(self.size_z, device='cuda', dtype=torch.float32)
        for i in range(self.clsN):
            self.T += self.Set[i].Calc_Memory()
        self.T = (self.T / self.clsN).detach()

    def get_DotLoss(self, z, y, reduction='mean', reverse=False):
        vectorSet = z - self.T
        if reverse:
            vectorSet = -vectorSet

        len_v = vectorSet.pow(2).sum(dim=1).sqrt()
        Dot = torch.sum(vectorSet * self.mean_v_Set[y], dim=1)
        Cosine = Dot/(len_v * self.len_v_Set[y])

        Zn = torch.abs((vectorSet - self.mean_v_Set[y])/self.sigma_v_Set[y])
        P = self.get_Gaussian_Percentage(Zn)
        
        if reverse:
            P = 1 - P

        loss = torch.sum((1 - Cosine) * P)

        if reduction == "mean":
            return loss / z.size(0)
        elif reduction == "sum":
            return loss

    def get_Gaussian_Percentage(self, Zn):
        # Scale.size = (Batch_size)
        P = torch.mean(self.Normal_Gaussian.cdf(Zn), dim=1) # 1-(P-0.5)*2 = 2-2P
        return 2-2*P

    def Calc_Pseudolabel(self, z, y):
        vectorSet = z - self.T
        cos = torch.nn.CosineSimilarity(dim=1)
        cos_result = torch.zeros((z.size(0), self.clsN), device="cuda", dtype=torch.float32)

        for i in range(self.clsN):
            mean_vector = self.mean_v_Set[i].unsqueeze(0).repeat(z.size(0), 1)
            cos_result[:, i] = cos(vectorSet, mean_vector)

        _, pseudo_label = cos_result.max(1)

        return pseudo_label.detach()

    def get_Regularizer(self, z):
        vectorSet = z - self.T
        len_v = vectorSet.pow(2).sum(dim=1).sqrt()
        s = torch.pow(torch.sum(len_v) / z.size(0), 2) # E(X)^2
        ss = torch.sum(len_v.pow(2)) / z.size(0)       # E(X^2)
        Regularizer = ss - s
        return Regularizer

    def Test_Init(self):
        for i in range(self.clsN):
            self.Set[i].Calc_Vector()
            self.mean_v_Set[i] = self.Set[i].mean_v.detach()
            self.len_v_Set[i] = self.Set[i].len_v.detach()
            self.sigma_v_Set[i] = self.Set[i].sigma_v.detach()

    def Calc_Test_Similarity(self, z, y):
        vectorSet = z - self.T
        Sim_scale = torch.tensor(0, device='cuda', dtype=torch.float32)
        Sim_vector = torch.tensor(0, device='cuda', dtype=torch.float32)

        cos = torch.nn.CosineSimilarity(dim=1)
        Sim_scale = torch.sum(torch.abs((vectorSet - self.mean_v_Set[y]))/self.sigma_v_Set[y]) / z.size(0)
        Sim_vector = torch.sum(torch.abs(cos(vectorSet, self.mean_v_Set[y])))

        return Sim_scale, Sim_vector


def train_NAE(args, state_info, Train_loader, Test_loader): # all 

    Memory = MemorySet(args=args)
    
    start_time = time.time()
    best_prec_result = torch.tensor(0, dtype=torch.float32)
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    mode = 'NAE'
    utils.default_model_dir = os.path.join(args.dir, mode)
    
    # criterion_BCE = torch.nn.BCELoss(reduction='mean')
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    checkpoint = utils.load_checkpoint(utils.default_model_dir)    
    if not checkpoint:
        args.last_epoch = -1
        state_info.learning_scheduler_init(args, mode)
    else:
        start_epoch = checkpoint['epoch'] + 1
        best_prec_result = checkpoint['Best_Prec']
        state_info.load_state_dict(checkpoint, mode)
        args.last_epoch = start_epoch
        state_info.learning_scheduler_init(args, mode)

    utils.print_log('Type, Epoch, Batch, total, Pseudo, Noise, Random, Regular, Noise%, Pseudo%')

    correct = torch.tensor(0, dtype=torch.float32)
    total_Size = torch.tensor(0, dtype=torch.float32)

    for it, (x, y, label) in enumerate(Train_loader):
        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)

        state_info.optim_NAE.zero_grad()
        z, c = state_info.forward_NAE(x)
        Memory.Batch_Insert(z, y)
        loss = criterion(c, y)
        loss.backward(retain_graph=True)
        state_info.optim_NAE.step()

        _, pred = torch.max(c.data, 1)
        correct += float(pred.eq(y.data).cpu().sum())
        total_Size += float(x.size(0))
        
        if it % 10 == 0:
            utils.print_log('Init, {}, {:.6f}, {:.3f}'
                  .format(it, loss.item(), 100.*correct / total_Size))
            print('Init, {}, {:.6f}, {:.3f}'
                  .format(it, loss.item(), 100.*correct / total_Size))

    for epoch in range(start_epoch, args.epoch):

        correct_Noise = torch.tensor(0, dtype=torch.float32)
        correct_Pseudo = torch.tensor(0, dtype=torch.float32)
        correct_Real = torch.tensor(0, dtype=torch.float32)
        correct_Test = torch.tensor(0, dtype=torch.float32)
        train_Size = torch.tensor(0, dtype=torch.float32)
        Pseudo_Noise = torch.tensor(0, dtype=torch.float32)
        Pseudo_Real = torch.tensor(0, dtype=torch.float32)

        # train
        state_info.NAE.train()
        utils.print_log('Type, Epoch, Batch, total, Cls, Noise_V, Random_V, Regular, Noise%, Pseudo%, Real%, Pseu_Nois%, Pseu_Real%')
        for it, (x, y, label) in enumerate(Train_loader):

            x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
            rand_y = torch.randint_like(y, low=0, high=10, device="cuda")

            state_info.optim_NAE.zero_grad()
            # with torch.autograd.set_detect_anomaly(True):
            z, c = state_info.forward_NAE(x)
            Memory.Batch_Insert(z, y)
            
            loss_N = Memory.get_DotLoss(z, y, reduction="mean", reverse=False)
            loss_R = Memory.get_DotLoss(z, rand_y, reduction="mean", reverse=True)
            reg = Memory.get_Regularizer(z)

            pseudo_label = Memory.Calc_Pseudolabel(z, y)
            loss = criterion(c, pseudo_label)

            total = args.t0 * loss + args.t1 * loss_N + args.t2 * loss_R + args.t3 * reg
            total.backward()
            state_info.optim_NAE.step()

            Pseudo_Noise += float(pseudo_label.eq(y).sum())
            Pseudo_Real += float(pseudo_label.eq(label).sum())
            _, pred = torch.max(c.data, 1)
            correct_Noise += float(pred.eq(y.data).cpu().sum())
            correct_Real += float(pred.eq(label.data).cpu().sum())
            correct_Pseudo += float(pred.eq(pseudo_label.data).cpu().sum())
            train_Size += float(x.size(0))

            if it % 10 == 0:
                utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'
                      .format(epoch, it, total.item(), loss.item(), loss_N.item(), loss_R.item(), reg.item()
                        , 100.*correct_Noise / train_Size
                        , 100.*correct_Pseudo / train_Size
                        , 100.*correct_Real / train_Size
                        , 100.*Pseudo_Noise / train_Size
                        , 100.*Pseudo_Real / train_Size))
                print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'
                      .format(epoch, it, total.item(), loss.item(), loss_N.item(), loss_R.item(), reg.item()
                        , 100.*correct_Noise / train_Size
                        , 100.*correct_Pseudo / train_Size
                        , 100.*correct_Real / train_Size
                        , 100.*Pseudo_Noise / train_Size
                        , 100.*Pseudo_Real / train_Size))

        testSize = torch.tensor(0, dtype=torch.float32)
        Similarity_Scale = torch.tensor(0, dtype=torch.float32)
        Similarity_Vector = torch.tensor(0, dtype=torch.float32)

        # test
        state_info.NAE.eval()
        Memory.Test_Init()
        for it, (x, y, label) in enumerate(Test_loader):

            x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)

            z, c = state_info.forward_NAE(x)
            Sim_scale, Sim_vector = Memory.Calc_Test_Similarity(z, y)

            Similarity_Scale += Sim_scale
            Similarity_Vector += Sim_vector

            _, pred = torch.max(c.data, 1)
            correct_Test += float(pred.eq(y.data).cpu().sum())
            testSize += float(x.size(0))

        utils.print_log('Type, Epoch, Batch, Scale, Vector, Percentage')

        utils.print_log('Test, {}, {}, {:.6f}, {:.6f}, {:.3f}'
              .format(epoch, it, Similarity_Scale / testSize, Similarity_Vector / testSize, 100.*correct_Test / testSize))
        print('Test, {}, {}, {:.6f}, {:.6f}, {:.3f}'
              .format(epoch, it, Similarity_Scale / testSize, Similarity_Vector / testSize, 100.*correct_Test / testSize))

        if 100.*correct_Test / testSize > best_prec_result:
            best_prec_result = 100.*correct_Test / testSize
            filename = 'checkpoint_best.pth.tar'
            utils.save_state_checkpoint(state_info, best_prec_result, epoch, mode, filename, utils.default_model_dir)

        filename = 'latest.pth.tar'
        utils.save_state_checkpoint(state_info, best_prec_result, epoch, mode, filename, utils.default_model_dir)
        state_info.lr_NAE.step()
        utils.print_log('')

    now = time.gmtime(time.time() - start_time)
    utils.print_log('Best Prec : {:.4f}'.format(best_prec_result.item()))
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))

# adversarial_loss = torch.nn.BCELoss()
# criterion_GAN = torch.nn.MSELoss()
# criterion_cycle = torch.nn.L1Loss()
# criterion_identity = torch.nn.L1Loss()
# criterion = nn.CrossEntropyLoss().cuda()