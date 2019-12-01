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

def print_time(start_time, log):
    print('{} : {}'.format(log, time.time() - start_time))

def print_time_relay(start_time, log):
    now = time.time()
    print('{} : {}'.format(log, now - start_time))
    return now

class Memory(object):
    def __init__(self, args):
        self.N = args.maxN # size of ALL Buffer
        self.index = 0
        self.index2 = 0
        self.z = torch.zeros([self.N, args.z], device="cuda", dtype=torch.float32)
        self.vector = torch.zeros([self.N, args.z], device="cuda", dtype=torch.float32)

    def Calc_Vector(self, eps=1e-9): # After 1 Epoch, it will calculated
        starttime = time.time()

        mean_len = self.vector.mean(dim=0).pow(2).sum().sqrt() + eps
        len_mean = self.vector.pow(2).sum(dim=1).sqrt().mean()
        self.mean_v = self.vector.mean(dim=0) * len_mean / mean_len
        self.sigma_v = self.vector.var(dim=0).sqrt()
        self.len_v = len_mean

        print_time(starttime, 'Memory : Calc_Vector')

    def Calc_Memory(self): # After 1 Epoch, it will calculated
        starttime = time.time()
        self.mean = self.z.mean(dim=0)
        self.sigma = self.z.var(dim=0).sqrt()
        print_time(starttime, 'Memory : Calc_Memory')
        return self.mean

    def Insert_memory(self, z): # Actual Function
        starttime = time.time()

        if self.index >= self.N:
            self.index = 0
        self.z[self.index] = z
        self.index = self.index + 1

        print_time(starttime, 'Memory : Insert_memory')
        
    def Insert_vector(self, vector): # Actual Function
        starttime = time.time()

        if self.index2 >= self.N:
            self.index2 = 0
        self.vector[self.index2] = vector
        self.index2 = self.index2 + 1

        print_time(starttime, 'Memory : Insert_vector')


class MemorySet(object):
    def __init__(self, args):
        self.clsN = args.clsN
        self.Set = []
        self.size_z = args.z
        for i in range(self.clsN):
            self.Set.append(Memory(args=args))

        self.mean_v_Set = torch.zeros((self.clsN, self.size_z), device="cuda", dtype=torch.float32)
        self.len_v_Set = torch.zeros((self.clsN), device="cuda", dtype=torch.float32)
        self.sigma_v_Set = torch.zeros((self.clsN, self.size_z), device="cuda", dtype=torch.float32)

    def Batch_Insert(self, z, y):

        starttime = time.time()

        for i in range(z.size(0)):
            Noise_label = y[i]
            data = z[i]
            self.Set[Noise_label].Insert_memory(data)

        starttime = print_time_relay(starttime, 'MemorySet : Batch_Insert, Num 1')

        self.Calc_Center()

        starttime = print_time_relay(starttime, 'MemorySet : Batch_Insert, Num 2')

        self.Batch_Vector_Insert(z, y)

        starttime = print_time_relay(starttime, 'MemorySet : Batch_Insert, Num 3')

        for i in range(self.clsN):
            self.Set[i].Calc_Vector()
            self.mean_v_Set[i] = self.Set[i].mean_v
            self.len_v_Set[i] = self.Set[i].len_v
            self.sigma_v_Set[i] = self.Set[i].sigma_v
        
        starttime = print_time_relay(starttime, 'MemorySet : Batch_Insert, Num 4')

    def Batch_Vector_Insert(self, z, y):
        starttime = time.time()

        vectorSet = z - self.T
        for i in range(vectorSet.size(0)):
            Noise_label = y[i]
            vector = vectorSet[i]
            self.Set[Noise_label].Insert_vector(vector)

        print_time(starttime, 'MemorySet : Batch_Vector_Insert')

    def Calc_Center(self):
        starttime = time.time()

        self.T = torch.zeros(self.size_z, device='cuda', dtype=torch.float32)
        for i in range(self.clsN):
            self.T += self.Set[i].Calc_Memory()

        self.T = self.T / self.clsN

        print_time(starttime, 'MemorySet : Calc_Center')







    def get_DotLoss(self, z, y, reduction='mean', reverse=False):
        starttime = time.time()

        vectorSet = z - self.T
        if reverse:
            vectorSet = -vectorSet

        len_v = vectorSet.pow(2).sum(dim=1).sqrt()
        Dot = torch.sum(VectorSet * self.mean_v_Set[y], dim=1)
        loss = torch.sum(len_v * self.len_v_Set[y] - Dot)
        
        # loss = torch.tensor(0, device='cuda', dtype=torch.float32)
        # for i in range(z.size(0)):
        #     label = y[i]
        #     vector = vectorSet[i]
        #     len_v = vector.pow(2).sum().sqrt()
        #     Dot = torch.sum(vector * self.Set[label].mean_v)
        #     loss += len_v * self.Set[label].len_v - Dot
        
        print_time(starttime, 'MemorySet : get_DotLoss')

        if reduction == "mean":
            return loss / z.size(0)
        elif reduction == "sum":
            return loss

    def get_Regularizer(self):
        starttime = time.time()

        s = torch.pow(torch.sum(self.len_v_Set) / self.clsN, 2) # E(X)^2
        ss = torch.sum(self.len_v_Set.pow(2)) / self.clsN       # E(X^2)

        # s = torch.tensor(0, device='cuda', dtype=torch.float32)
        # ss = torch.tensor(0, device='cuda', dtype=torch.float32)
        # for i in range(self.clsN):
        #     s += self.Set[i].len_v
        #     ss += self.Set[i].len_v.pow(2)

        # s = (s / self.clsN).pow(2)   # E(X)^2
        # ss = ss / self.clsN # E(X^2)

        Regularizer = ss - s
        
        print_time(starttime, 'MemorySet : get_Regularizer')

        return Regularizer


    def Test_Init(self):

        starttime = time.time()

        for i in range(self.clsN):
            self.Set[i].Calc_Vector()
            self.mean_v_Set[i] = self.Set[i].mean_v
            self.len_v_Set[i] = self.Set[i].len_v
            self.sigma_v_Set[i] = self.Set[i].sigma_v

        print_time(starttime, 'MemorySet : Test_Init')

    def Calc_Test_Similarity(self, z, y):
        starttime = time.time()

        vectorSet = z - self.T
        Sim_scale = torch.tensor(0, device='cuda', dtype=torch.float32)
        Sim_vector = torch.tensor(0, device='cuda', dtype=torch.float32)

        cos = torch.nn.CosineSimilarity(dim=1)
        Sim_scale = torch.sum((VectorSet - self.mean_v_Set[y])/self.sigma_v_Set[y])
        Sim_vector = torch.sum(torch.abs(cos(VectorSet, self.mean_v_Set[y])))

        # for i in range(z.size(0)):
        #     label = y[i]
        #     vector = vectorSet[i]

        #     Sim_scale += torch.sum((vector - self.Set[label].mean_v) / self.Set[label].sigma_v)
        #     Sim_vector += cos(vector, self.Set[label].mean_v)

        print_time(starttime, 'MemorySet : Calc_Test_Similarity')

        return Sim_scale, Sim_vector


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

    utils.print_log('Type, Epoch, Batch, total, Recon, Noise, Random, Regular')

    # Init Learning
    for it, (x, y, label) in enumerate(Train_loader):

        inittime = time.time()

        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
        rand_y = torch.randint_like(y, low=0, high=10, device="cuda")

        state_info.optim_NAE.zero_grad()
        
        z, x_h = state_info.forward_NAE(x)
        Memory.Batch_Insert(z, y)
        loss = criterion_BCE(x_h, x)
        loss.backward(retain_graph=True)

        state_info.optim_NAE.step()

        if it % 10 == 0:
            utils.print_log('Init, {}, {:.6f}'
                  .format(it, loss.item()))
            print('Init, {}, {:.6f}'
                  .format(it, loss.item()))

        print_time(inittime, 'Init main : Num 8')


        if it>40:
            break;

    for epoch in range(start_epoch, args.epoch):

        # train
        state_info.NAE.train()
        for it, (x, y, label) in enumerate(Train_loader):

            inittime = time.time()

            x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
            rand_y = torch.randint_like(y, low=0, high=10, device="cuda")

            state_info.optim_NAE.zero_grad()

            starttime = time.time()

            z, x_h = state_info.forward_NAE(x)
            
            starttime = print_time_relay(starttime, 'Main : Num 1')

            Memory.Batch_Insert(z, y)

            starttime = print_time_relay(starttime, 'Main : Num 2')

            loss_N = Memory.get_DotLoss(z, y, reduction="mean", reverse=False)

            starttime = print_time_relay(starttime, 'Main : Num 3')

            loss_R = Memory.get_DotLoss(z, rand_y, reduction="mean", reverse=True)

            starttime = print_time_relay(starttime, 'Main : Num 4')

            reg = Memory.get_Regularizer()

            starttime = print_time_relay(starttime, 'Main : Num 5')

            loss = criterion_BCE(x_h, x)

            starttime = print_time_relay(starttime, 'Main : Num 6')

            # total = loss + args.t1 * loss_N + args.t2 * loss_R + args.t3 * reg
            # total.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            t = print_time_relay(starttime, 'Main : Loss1')
            loss_N.backward(retain_graph=True)
            t = print_time_relay(t, 'Main : Loss Noise')
            loss_R.backward(retain_graph=True)
            t = print_time_relay(t, 'Main : Loss Random')
            reg.backward(retain_graph=True)
            t = print_time_relay(t, 'Main : Regularizer')

            starttime = print_time_relay(starttime, 'Main : Num 7')


            state_info.optim_NAE.step()

            print_time(inittime, 'Main : Num 8')

            break;

            if it % 10 == 0:
                utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'
                      .format(epoch, it, total.item(), loss.item(), loss_N.item(), loss_R.item(), reg.item()))
                print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'
                      .format(epoch, it, total.item(), loss.item(), loss_N.item(), loss_R.item(), reg.item()))

        testSize = torch.tensor(0, dtype=torch.float32)
        Similarity_Scale = torch.tensor(0, dtype=torch.float32)
        Similarity_Vector = torch.tensor(0, dtype=torch.float32)

        # test
        state_info.NAE.eval()
        Memory.Test_Init()
        for it, (x, y, label) in enumerate(Test_loader):

            x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)

            z, x_h = state_info.forward_NAE(x)
            Sim_scale, Sim_vector = Memory.Calc_Test_Similarity(z, y)

            Similarity_Scale += Sim_scale
            Similarity_Vector += Sim_vector

            testSize += float(x.size(0))

        utils.print_log('Type, Epoch, Batch, Scale, Vector')

        utils.print_log('Test, {}, {}, {:.6f}, {:.6f}'
              .format(epoch, it, Similarity_Scale / testSize, Similarity_Vector / testSize))
        print('Test, {}, {}, {:.6f}, {:.6f}'
              .format(epoch, it, Similarity_Scale / testSize, Similarity_Vector / testSize))

        Generation(args, state_info, Memory, epoch)

        filename = 'latest.pth.tar'
        utils.save_state_checkpoint(state_info, best_prec_result, epoch, mode, filename, utils.default_model_dir)
        state_info.lr_NAE.step()
        utils.print_log('')

    now = time.gmtime(time.time() - start_time)
    utils.print_log('Best Prec : {:.4f}'.format(best_prec_result.item()))
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))

def Generation(args, state_info, Memory, epoch):

    img_path = utils.make_directory(os.path.join(utils.default_model_dir, 'images'))

    Rand = torch.rand(10)
    ImageSet = []
    for i in range(args.clsN):
        r = Rand[i]
        z = Memory.Set[i].sigma_v * r + Memory.Set[i].mean_v + Memory.T
        x_h = state_info.test_NAE(z.view(1,-1))
        ImageSet.append(x_h.view(1,-1,32,32))

    z = Memory.T
    x_h = state_info.test_NAE(z.view(1,-1))
    ImageSet.append(x_h.view(1,-1,32,32))
    ImageSet.append(x_h.view(1,-1,32,32))

    ImageSet = torch.cat(ImageSet, dim=0)
    merge = merge_images(to_data(ImageSet))

    save_image(merge.data, os.path.join(img_path, '%d.png' % epoch), normalize=True)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def merge_images(imageSet, row=3, col=4):
    _, _, h, w = imageSet.shape
    merged = np.zeros([1, row*h, col*w])
    for idx, (s) in enumerate(imageSet):
        i = idx // row
        j = idx % col
        if i is row:
            break
        merged[:, i*h:(i+1)*h, (j)*w:(j+1)*w] = s

    return torch.from_numpy(merged)

# adversarial_loss = torch.nn.BCELoss()
# criterion_GAN = torch.nn.MSELoss()
# criterion_cycle = torch.nn.L1Loss()
# criterion_identity = torch.nn.L1Loss()
# criterion = nn.CrossEntropyLoss().cuda()