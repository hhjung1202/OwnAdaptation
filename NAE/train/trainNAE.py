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

def to_var(x, dtype):
    return Variable(x.type(dtype))

def get_percentage_Fake(Fake_loader):

    correct = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)

    for it, (fake, Fy, label_Fy) in enumerate(Fake_loader):
        resultF = label_Fy.eq(Fy).cpu().type(torch.ByteTensor)
        correct += float(resultF.sum())
        total += float(fake.size(0))
    
    print('Fake Dataset Percentage', 100. * correct / total)
    return 100. * correct / total


# args.maxN
# args.alpha weight Upsample
# args.beta cos simm param
class Memory(object):
    def __init__(self, args):
        self.N = args.maxN # size of ALL [max 제한 둬야함]
        self.index = 0
        self.index2 = 0
        self.z = torch.zeros([self.N, args.z], device="cuda", dtype=torch.float32)
        self.vector = torch.zeros([self.N, args.z], device="cuda", dtype=torch.float32)
        # self.beta = args.beta
        # self.alpha = args.alpha
        # Index를 두어 최신화 시킬 Index를 Update한다. 1 2 3 4 5 1 2 3 4 5 (이런식으로 Not Push Pop, Circular Queue)

    def Calc_Vector(self): # After 1 Epoch, it will calculated
        self.mean_v = self.vector.mean(dim=0)
        self.var_v = self.vector.var(dim=0)
        self.len_v = self.mean_v.pow(2).sum().sqrt()

    def Calc_Memory(self): # After 1 Epoch, it will calculated
        self.mean = self.z.mean(dim=0)
        self.var = self.z.var(dim=0)
        return self.mean

    def Insert_memory(self, z): # Actual Function
        if self.index >= self.N:
            self.index = 0
        self.z[self.index] = z
        self.index = self.index + 1
        
    def Insert_vector(self, vector): # Actual Function
        if self.index2 >= self.N:
            self.index2 = 0
        self.vector[self.index2] = vector
        self.index2 = self.index2 + 1

    # def Calc_CosSim_N(self): # After 1 Epoch, it will calculated

    #     cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #     mu_repeat = self.mean.repeat([self.N, 1])
    #     simm = cos(self.z ,mu_repeat)
    #     self.n = int(torch.sum(simm > self.beta))
    #     # Cos Similarity 를 쓰지 않고 Normalization을 사용해도 된다.
    #     # Variance를 고려해야만 한다. 하나의 돌출변수로 인한 Cosine Sim 변화율이 너무 크다.


    # def Weighted_Mean(self): # Actual Function, After 1 Epoch, it will calculated
    #     self.Calc_Memory()


    #     return self.mean * self.alpha

        # if self.n <= 0.1 * self.N:
        #     return self.mean
        # elif self.n <= 0.4 * self.N:
        #     return self.mean * 3
        # else:
        #     return self.mean * 9*self.N / (10*self.n - self.N)


class MemorySet(object):
    def __init__(self, args):
        self.clsN = args.clsN
        self.Set = []
        for i in range(self.clsN):
            self.Set.append(Memory(args=args))

    def Batch_Insert(self, z, y):

        for i in range(z.size(0)):
            Noise_label = y[i]
            data = z[i]
            self.Set[Noise_label].Insert_memory(data)

        self.Calc_Center()
        self.Batch_Vector_Insert(z, y)

        for i in range(self.clsN):
            self.Set[i].Calc_Vector()

    def Batch_Vector_Insert(self, z, y):
        vector = z - self.T
        for i in range(vector.size(0)):
            Noise_label = y[i]
            data = vector[i]
            self.Set[Noise_label].Insert_vector(data)

    def Calc_Center(self):
        self.T = None
        for i in range(self.clsN):
            if self.T is None:
                self.T = self.Set[i].Calc_Memory()
            else:
                self.T = self.T + self.Set[i].Calc_Memory()

        self.T = self.T / self.clsN

    def get_DotLoss(self, z, y, reduction='mean', reverse=False):
        vector = z - self.T
        if reverse:
            vector = -vector
        
        loss = None
        for i in range(z.size(0)):
            label = y[i]
            data = vector[i]
            len_v = data.pow(2).sum().sqrt()
            Dot = torch.sum(data * self.Set[label].mean_v)
            if loss is None:
                loss = len_v * self.Set[label].len_v - Dot
            else:
                loss += len_v * self.Set[label].len_v - Dot
        if reduction == "mean":
            return loss / z.size(0)
        elif reduction == "sum":
            return loss

    def get_Regularizer(self):
        s = None
        ss = None
        for i in range(self.clsN):
            if s is None:
                s = self.Set[i].len_v
                ss = self.Set[i].len_v.pow(2)
            else:
                s += self.Set[i].len_v
                ss += self.Set[i].len_v.pow(2)

        s = (s / self.clsN).pow(2)   # E(X)^2
        ss = ss / self.clsN # E(X^2)

        Regularizer = ss - s
        return Regularizer

def train_NAE(args, state_info, Train_loader, Test_loader): # all 

    Memory = MemorySet(args=args)
    
    best_prec_result = torch.tensor(0, dtype=torch.float32)
    start_time = time.time()
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    mode = 'NAE'
    utils.default_model_dir = os.path.join(args.dir, mode)
    
    criterion_BCE = torch.nn.BCELoss(reduction='mean')
    # criterion = torch.nn.CrossEntropyLoss()

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

    utils.print_log('Type, Epoch, Batch, loss, BCE, KLD, CE')

    for epoch in range(start_epoch, args.epoch):

        correct_Noise = torch.tensor(0, dtype=torch.float32)
        correct_Real = torch.tensor(0, dtype=torch.float32)
        correct_Test = torch.tensor(0, dtype=torch.float32)
        total = torch.tensor(0, dtype=torch.float32)

        # train
        state_info.NAE.train()
        for it, (x, y, label) in enumerate(Train_loader):

            x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
            rand_y = torch.randint_like(y, low=0, high=10, device="cuda")

            state_info.optim_NAE.zero_grad()
            
            z, x_h = state_info.forward_NAE(x)
            Memory.Batch_Insert(z, y)
            loss_N = Memory.get_DotLoss(z, y, reduction="mean", reverse=False)
            loss_R = Memory.get_DotLoss(z, rand_y, reduction="mean", reverse=True)
            reg = Memory.get_Regularizer()
            loss = criterion_BCE(x_h, x)
            total = loss + args.t1 * loss_N + args.t2 * loss_R + args.t3 * reg
            total.backward()

            state_info.optim_NAE.step()

            if it % 10 == 0:
                utils.print_log('main Train, {}, {}, {:.6f}, {:.3f}, {:.3f}'
                      .format(epoch, it, loss.item(), 100.*correct_Noise / total, 100.*correct_Real / total))
                print('main Train, {}, {}, {:.6f}, {:.3f}, {:.3f}'
                      .format(epoch, it, loss.item(), 100.*correct_Noise / total, 100.*correct_Real / total))

        total = torch.tensor(0, dtype=torch.float32)
        # test
        state_info.NAE.eval()
        for it, (Test, Ty, label_Ty) in enumerate(Test_loader):

            Test, Ty, label_Ty = to_var(Test, FloatTensor), to_var(Ty, LongTensor), to_var(label_Ty, LongTensor)

            cls_out = state_info.forward_NAE(Test)

            _, pred = torch.max(cls_out.data, 1)
            correct_Test += float(pred.eq(Ty.data).cpu().sum())
            total += float(Noise.size(0))

        utils.print_log('main Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correct_Test / total))
        print('main Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correct_Test / total))

        if 100.*correct_Test / total > best_prec_result:
            best_prec_result = 100.*correct_Test / total
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