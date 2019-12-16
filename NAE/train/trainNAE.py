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
import torch.nn.functional as F
import torch.distributions.normal as normal
from .Memory import MemorySet

def to_var(x, dtype):
    return Variable(x.type(dtype))

class NegativeCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(NegativeCrossEntropyLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, Out, y):
        Negative_log_P = torch.log(1 - self.softmax(Out))
        return F.nll_loss(Negative_log_P, y) # y is Random label    

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
    softmax = torch.nn.Softmax(dim=1)
    Neg_criterion = NegativeCrossEntropyLoss()

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

    # for it, (x, y, label) in enumerate(Train_loader):
    #     x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)

    #     state_info.optim_NAE.zero_grad()
    #     z, c = state_info.forward_NAE(x)
    #     Memory.Batch_Insert(z, y)
    #     loss = criterion(c, y)
    #     loss.backward(retain_graph=True)
    #     state_info.optim_NAE.step()

    #     _, pred = torch.max(c.data, 1)
    #     correct += float(pred.eq(y.data).cpu().sum())
    #     total_Size += float(x.size(0))
        
    #     if it % 10 == 0:
    #         utils.print_log('Init, {}, {:.6f}, {:.3f}'
    #               .format(it, loss.item(), 100.*correct / total_Size))
    #         print('Init, {}, {:.6f}, {:.3f}'
    #               .format(it, loss.item(), 100.*correct / total_Size))

    for epoch in range(start_epoch, args.epoch):

        correct_Noise = torch.tensor(0, dtype=torch.float32)
        correct_Pseudo = torch.tensor(0, dtype=torch.float32)
        correct_Real = torch.tensor(0, dtype=torch.float32)
        correct_Test = torch.tensor(0, dtype=torch.float32)
        train_Size = torch.tensor(0, dtype=torch.float32)
        Pseudo_Noise = torch.tensor(0, dtype=torch.float32)
        Pseudo_Real = torch.tensor(0, dtype=torch.float32)

        state_info.NAE.train()
        utils.print_log('Type, Epoch, Batch, loss, Real%, Noise%')
        for it, (x, y, label) in enumerate(Train_loader):
            x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)

            state_info.optim_NAE.zero_grad()
            _, c = state_info.forward_NAE(x)
            loss = criterion(c, y)
            loss.backward(retain_graph=True)
            state_info.optim_NAE.step()

            _, pred = torch.max(c.data, 1)
            correct_Real += float(pred.eq(label.data).cpu().sum())
            correct_Noise += float(pred.eq(y.data).cpu().sum())
            train_Size += float(x.size(0))
            
            if it % 10 == 0:
                utils.print_log('Init, {}, {:.6f}, {:.3f}, {:.3f}, {:.3f}'
                      .format(epoch, it, loss.item(), 100.*correct_Real / train_Size, 100.*correct_Noise / train_Size))
                print('Init, {}, {:.6f}, {:.3f}, {:.3f}, {:.3f}'
                      .format(epoch, it, loss.item(), 100.*correct_Real / train_Size, 100.*correct_Noise / train_Size))

        # train
        # state_info.NAE.train()
        # utils.print_log('Type, Epoch, Batch, total, Cls, Noise_V, Random_V, Regular, Noise%, Pseudo%, Real%, Pseu_Nois%, Pseu_Real%')
        # for it, (x, y, label) in enumerate(Train_loader):

        #     x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
        #     rand_y = torch.randint_like(y, low=0, high=10, device="cuda")

        #     state_info.optim_NAE.zero_grad()
        #     # with torch.autograd.set_detect_anomaly(True):
        #     z, c = state_info.forward_NAE(x)
        #     Memory.Batch_Insert(z, y)

        #     loss_N = Memory.get_DotLoss_Noise(z, y, reduction="mean", reverse=False)
        #     loss_R = Memory.get_DotLoss_Noise(z, rand_y, reduction="mean", reverse=True)    
        #     # loss_N = Memory.get_DotLoss_BASE(z, y, reduction="mean", reverse=False)
        #     # loss_R = Memory.get_DotLoss_BASE(z, rand_y, reduction="mean", reverse=True)    
            
        #     reg = Memory.get_Regularizer(z)
        #     pseudo_label = Memory.Calc_Pseudolabel(z, y)
        #     loss = criterion(c, pseudo_label)

        #     total = args.t0 * loss + args.t1 * loss_N + args.t2 * loss_R + args.t3 * reg    
        #     total.backward()
        #     state_info.optim_NAE.step()

        #     Pseudo_Noise += float(pseudo_label.eq(y).sum())
        #     Pseudo_Real += float(pseudo_label.eq(label).sum())
        #     _, pred = torch.max(c.data, 1)
        #     correct_Noise += float(pred.eq(y.data).cpu().sum())
        #     correct_Real += float(pred.eq(label.data).cpu().sum())
        #     correct_Pseudo += float(pred.eq(pseudo_label.data).cpu().sum())
        #     train_Size += float(x.size(0))

        #     if it % 10 == 0:
        #         utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'
        #               .format(epoch, it, total.item(), loss.item(), loss_N.item(), loss_R.item(), reg.item()
        #                 , 100.*correct_Noise / train_Size
        #                 , 100.*correct_Pseudo / train_Size
        #                 , 100.*correct_Real / train_Size
        #                 , 100.*Pseudo_Noise / train_Size
        #                 , 100.*Pseudo_Real / train_Size))
        #         print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'
        #               .format(epoch, it, total.item(), loss.item(), loss_N.item(), loss_R.item(), reg.item()
        #                 , 100.*correct_Noise / train_Size
        #                 , 100.*correct_Pseudo / train_Size
        #                 , 100.*correct_Real / train_Size
        #                 , 100.*Pseudo_Noise / train_Size
        #                 , 100.*Pseudo_Real / train_Size))

        testSize = torch.tensor(0, dtype=torch.float32)
        Similarity_Scale = torch.tensor(0, dtype=torch.float32)
        Similarity_Vector = torch.tensor(0, dtype=torch.float32)

        # test
        state_info.NAE.eval()
        # Memory.Test_Init()
        for it, (x, y, label) in enumerate(Test_loader):

            x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)

            z, c = state_info.forward_NAE(x)
            # Sim_scale, Sim_vector = Memory.Calc_Test_Similarity(z, y)
            # 
            # Similarity_Scale += Sim_scale
            # Similarity_Vector += Sim_vector

            _, pred = torch.max(c.data, 1)
            correct_Test += float(pred.eq(y.data).cpu().sum())
            testSize += float(x.size(0))

        utils.print_log('Type, Epoch, Batch, Scale, Vector, Percentage')

        utils.print_log('Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correct_Test / testSize))
        print('Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correct_Test / testSize))

        # utils.print_log('Test, {}, {}, {:.6f}, {:.6f}, {:.3f}'
        #       .format(epoch, it, Similarity_Scale / testSize, Similarity_Vector / testSize, 100.*correct_Test / testSize))
        # print('Test, {}, {}, {:.6f}, {:.6f}, {:.3f}'
        #       .format(epoch, it, Similarity_Scale / testSize, Similarity_Vector / testSize, 100.*correct_Test / testSize))

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