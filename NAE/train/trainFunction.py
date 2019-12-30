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

def train_step1(state_info, Train_loader, Test_loader, Memory, criterion, epoch):

    state_info.model.train()
	utils.print_log('Type, Epoch, Batch, total, percentage')
    correct = torch.tensor(0, dtype=torch.float32)
    total_Size = torch.tensor(0, dtype=torch.float32)

    for it, (x, y, label) in enumerate(Train_loader):
        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)

        state_info.optim_model.zero_grad()
        out, z = state_info.forward(x)
        Memory.Batch_Insert(z, y)
        loss = criterion(out, y)
        loss.backward(retain_graph=True)
        state_info.optim_model.step()

        _, pred = torch.max(out.data, 1)
        correct += float(pred.eq(y.data).cpu().sum())
        total_Size += float(x.size(0))
        
        if it % 10 == 0:
            utils.print_log('Init, {}, {}, {:.6f}, {:.3f}'
                  .format(epoch, it, loss.item(), 100.*correct / total_Size))
            print('Init, {}, {}, {:.6f}, {:.3f}'
                  .format(epoch, it, loss.item(), 100.*correct / total_Size))

    epoch_result = test(state_info, Test_loader, epoch)
    return epoch_result


def train_step2(state_info, Train_loader, Test_loader, Memory, criterion, epoch):
	correct_Noise = torch.tensor(0, dtype=torch.float32)
    correct_Pseudo = torch.tensor(0, dtype=torch.float32)
    correct_Real = torch.tensor(0, dtype=torch.float32)
    train_Size = torch.tensor(0, dtype=torch.float32)
    Pseudo_Noise = torch.tensor(0, dtype=torch.float32)
    Pseudo_Real = torch.tensor(0, dtype=torch.float32)

    # train
    state_info.model.train()
    utils.print_log('Type, Epoch, Batch, total, Cls, Noise_V, Random_V, Regular, Noise%, Pseudo%, Real%, Pseu_Nois%, Pseu_Real%')
    for it, (x, y, label) in enumerate(Train_loader):

        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
        rand_y = torch.randint_like(y, low=0, high=10, device="cuda")

        state_info.optim_model.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
        out, z = state_info.forward(x)
        Memory.Batch_Insert(z, y)

        loss_N = Memory.get_DotLoss_Noise(z, y, reduction="mean", reverse=False)
        loss_R = Memory.get_DotLoss_Noise(z, rand_y, reduction="mean", reverse=True)    

        reg = Memory.get_Regularizer(z)
        pseudo_label = Memory.Calc_Pseudolabel(z, y)
        loss = criterion(out, y)

        total = args.t0 * loss + args.t1 * loss_N + args.t2 * loss_R + args.t3 * reg    
        total.backward()
        state_info.optim_model.step()

        Pseudo_Noise += float(pseudo_label.eq(y).sum())
        Pseudo_Real += float(pseudo_label.eq(label).sum())
        _, pred = torch.max(out.data, 1)
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

    epoch_result = test(state_info, Test_loader, epoch)
    return epoch_result


def train_step3(state_info, Train_loader, Test_loader, Memory, criterion, epoch):
	correct_Noise = torch.tensor(0, dtype=torch.float32)
    correct_Pseudo = torch.tensor(0, dtype=torch.float32)
    correct_Real = torch.tensor(0, dtype=torch.float32)
    train_Size = torch.tensor(0, dtype=torch.float32)
    Pseudo_Noise = torch.tensor(0, dtype=torch.float32)
    Pseudo_Real = torch.tensor(0, dtype=torch.float32)

    # train
    state_info.model.train()
    utils.print_log('Type, Epoch, Batch, total, Cls, Noise_V, Random_V, Regular, Noise%, Pseudo%, Real%, Pseu_Nois%, Pseu_Real%')
    for it, (x, y, label) in enumerate(Train_loader):

        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
        rand_y = torch.randint_like(y, low=0, high=10, device="cuda")

        state_info.optim_model.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
        out, z = state_info.forward(x)
        Memory.Batch_Insert(z, y)

        loss_N = Memory.get_DotLoss_Noise(z, y, reduction="mean", reverse=False)
        loss_R = Memory.get_DotLoss_Noise(z, rand_y, reduction="mean", reverse=True)    

        reg = Memory.get_Regularizer(z)
        pseudo_label = Memory.Calc_Pseudolabel(z, y)
        loss = criterion(out, pseudo_label)

        total = args.t0 * loss + args.t1 * loss_N + args.t2 * loss_R + args.t3 * reg    
        total.backward()
        state_info.optim_model.step()

        Pseudo_Noise += float(pseudo_label.eq(y).sum())
        Pseudo_Real += float(pseudo_label.eq(label).sum())
        _, pred = torch.max(out.data, 1)
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
    
    epoch_result = test(state_info, Test_loader, epoch)
    return epoch_result

def test(state_info, Test_loader, epoch):
	
	testSize = torch.tensor(0, dtype=torch.float32)
    Similarity_Scale = torch.tensor(0, dtype=torch.float32)
    Similarity_Vector = torch.tensor(0, dtype=torch.float32)
    correct_Test = torch.tensor(0, dtype=torch.float32)

    # test
    state_info.model.eval()
    for it, (x, y) in enumerate(Test_loader):

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)

        out, z = state_info.forward(x)

        _, pred = torch.max(out.data, 1)
        correct_Test += float(pred.eq(y.data).cpu().sum())
        testSize += float(x.size(0))

    utils.print_log('Type, Epoch, Batch, Percentage')

    utils.print_log('Test, {}, {}, {:.3f}'
          .format(epoch, it, 100.*correct_Test / testSize))
    print('Test, {}, {}, {:.3f}'
          .format(epoch, it, 100.*correct_Test / testSize))

    return 100.*correct_Test / testSize