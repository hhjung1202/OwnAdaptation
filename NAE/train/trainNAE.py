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

def Loss_Dot():
    pass

def Reg_vector():
    pass

def c(n, N):
    return 9*N / (10*n-N)

class Memory(object):
    def __init__(self):
        self.N = 0 # size of ALL [max 제한 둬야함]

        pass 
        # Torch Tensor로 Concat하면서 CUDA 메모리에 축적하고, 만약 너무 크다면 CPU 메모리로 변환한 후 Concat한다.
        # Index를 두어 최신화 시킬 Index를 Update한다. 1 2 3 4 5 1 2 3 4 5 (이런식으로 Not Push Pop, Circular Queue)


    def get_Mean_Var(self):
        # return mean, var
        pass

    def Insert(self):
        pass

    def Calc_CosSim_N(self):
        self.n = 0 # size of Similar vector
        pass
        # Cos Similarity 를 쓰지 않고 Normalization을 사용해도 된다.

    def Weighted_Mean(self):
        if self.n <= 0.1 * self.N:
            return self.mean
        elif self.n <= 0.4 * self.N:
            return self.mean * 3
        else:
            return self.mean * 9*self.N / (10*self.n - self.N)


class MemorySet(object):
    def __init__(self):
        self.Set = {
            0: Memory(),
            1: Memory(),
            2: Memory(),
            3: Memory(),
            4: Memory(),
            5: Memory(),
            6: Memory(),
            7: Memory(),
            8: Memory(),
            9: Memory(),
        }

    def get_Center(self):
        Sum = None
        for i in range(10):
            if Sum is None:
                Sum = self.Set[i].Weighted_Mean()
            else:
                Sum = Sum + self.Set[i].Weighted_Mean()

        return Sum / 10


def train_NAE(args, state_info, All_loader, Test_loader): # all 
    
    best_prec_result = torch.tensor(0, dtype=torch.float32)
    start_time = time.time()
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    mode = 'NAE'
    utils.default_model_dir = os.path.join(args.dir, mode)
    
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

    utils.print_log('Type, Epoch, Batch, loss, BCE, KLD, CE')
    state_info.set_train_mode()

    for epoch in range(start_epoch, args.epoch):

        correct_Noise = torch.tensor(0, dtype=torch.float32)
        correct_Real = torch.tensor(0, dtype=torch.float32)
        correct_Test = torch.tensor(0, dtype=torch.float32)
        total = torch.tensor(0, dtype=torch.float32)

        # train
        state_info.NAE.train()
        for it, (All, Ay, label_Ay) in enumerate(Noise_loader):

            All, Ay, label_Ay = to_var(All, FloatTensor), to_var(Ay, LongTensor), to_var(label_Ay, LongTensor)

            cls_out = state_info.forward_NAE(All)

            state_info.optim_NAE.zero_grad()
            loss = criterion(cls_out, Ay)
            loss.backward()
            state_info.optim_NAE.step()

            _, pred = torch.max(cls_out.data, 1)
            correct_Noise += float(pred.eq(Ay.data).cpu().sum())
            correct_Real += float(pred.eq(label_Ay.data).cpu().sum())
            total += float(All.size(0))
            
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