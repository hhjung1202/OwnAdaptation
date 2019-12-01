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
        self.z[self.index] = z
        self.index = self.index + 1
        
    def Insert_vector(self, vector): # Actual Function
        if self.index2 >= self.N:
            self.index2 = 0
        self.vector[self.index2] = vector
        self.index2 = self.index2 + 1


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
        vectorSet = z - self.T
        for i in range(vectorSet.size(0)):
            Noise_label = y[i]
            vector = vectorSet[i]
            self.Set[Noise_label].Insert_vector(vector)

    def Calc_Center(self):
        self.T = None
        for i in range(self.clsN):
            if self.T is None:
                self.T = self.Set[i].Calc_Memory()
            else:
                self.T = self.T + self.Set[i].Calc_Memory()

        self.T = self.T / self.clsN

    def get_DotLoss(self, z, y, reduction='mean', reverse=False):
        vectorSet = z - self.T
        if reverse:
            vectorSet = -vectorSet
        
        loss = torch.tensor(0, device='cuda', dtype=torch.float32)
        for i in range(z.size(0)):
            label = y[i]
            vector = vectorSet[i]
            len_v = vector.pow(2).sum().sqrt()
            Dot = torch.sum(vector * self.Set[label].mean_v)
            loss += len_v * self.Set[label].len_v - Dot
        if reduction == "mean":
            return loss / z.size(0)
        elif reduction == "sum":
            return loss

    def get_Regularizer(self):
        s = torch.tensor(0, device='cuda', dtype=torch.float32)
        ss = torch.tensor(0, device='cuda', dtype=torch.float32)
        for i in range(self.clsN):
            s += self.Set[i].len_v
            ss += self.Set[i].len_v.pow(2)

        s = (s / self.clsN).pow(2)   # E(X)^2
        ss = ss / self.clsN # E(X^2)

        Regularizer = ss - s
        return Regularizer

    def Calc_Test_Similarity(self, z, y):
        vectorSet = z - self.T
        Sim_scale = torch.tensor(0, device='cuda', dtype=torch.float32)
        Sim_vector = torch.tensor(0, device='cuda', dtype=torch.float32)
        cos = torch.nn.CosineSimilarity(dim=0)
        for i in range(z.size(0)):
            label = y[i]
            vector = vectorSet[i]

            Sim_scale += torch.sum((vector - self.Set[label].mean_v) / self.Set[label].sigma_v)
            Sim_vector += cos(vector, self.Set[label].mean_v)

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

        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
        rand_y = torch.randint_like(y, low=0, high=10, device="cuda")

        state_info.optim_NAE.zero_grad()
        
        z, x_h = state_info.forward_NAE(x)
        with torch.autograd.detect_anomaly():
            Memory.Batch_Insert(z, y)
            loss = criterion_BCE(x_h, x)
            loss.backward(retain_graph=True)

        state_info.optim_NAE.step()

        if it % 10 == 0:
            utils.print_log('Init, {}, {:.6f}'
                  .format(it, loss.item()))
            print('Init, {}, {:.6f}'
                  .format(it, loss.item()))

    for epoch in range(start_epoch, args.epoch):

        # train
        state_info.NAE.train()
        for it, (x, y, label) in enumerate(Train_loader):

            x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
            rand_y = torch.randint_like(y, low=0, high=10, device="cuda")

            state_info.optim_NAE.zero_grad()
            
            z, x_h = state_info.forward_NAE(x)
            with torch.autograd.detect_anomaly():
                Memory.Batch_Insert(z, y)
                loss_N = Memory.get_DotLoss(z, y, reduction="mean", reverse=False)
                loss_R = Memory.get_DotLoss(z, rand_y, reduction="mean", reverse=True)
                reg = Memory.get_Regularizer()
                loss = criterion_BCE(x_h, x)
                total = loss + args.t1 * loss_N + args.t2 * loss_R + args.t3 * reg
                total.backward(retain_graph=True)

            state_info.optim_NAE.step()

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
        z = Memory.Set[i].sigma_v * r + self.Set[i].mean_v + Memory.T
        x_h = state_info.test_NAE(z)
        ImageSet.append(x_h.view(1,-1,32,32))

    z = Memory.T
    x_h = state_info.test_NAE(z)
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