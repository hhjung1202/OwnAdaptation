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

def train_disc(state_info, True_loader, Fake_loader, Noise_Test_loader, start_epoch, last_epoch): # all 
    
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    criterion_GAN = torch.nn.BCELoss()

    # criterion_GAN = torch.nn.MSELoss()

    utils.print_log('Type, Epoch, Batch, loss, BCE, KLD, CE')

    percentage = get_percentage_Fake(Fake_loader)

    valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
    unvalid = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

    for epoch in range(start_epoch, last_epoch):

        correctR = torch.tensor(0, dtype=torch.float32)
        correctF = torch.tensor(0, dtype=torch.float32)
        correctN = torch.tensor(0, dtype=torch.float32)
        total = torch.tensor(0, dtype=torch.float32)

        # train
        state_info.disc.train()
        for it, ((real, Ry, label_Ry), (fake, Fy, label_Fy)) in enumerate(zip(True_loader, Fake_loader)):

            real, Ry, label_Ry = to_var(real, FloatTensor), to_var(Ry, LongTensor), to_var(label_Ry, LongTensor)
            fake, Fy, label_Fy = to_var(fake, FloatTensor), to_var(Fy, LongTensor), to_var(label_Fy, LongTensor)

            Rout, Fout = state_info.forward_disc(real, Ry), state_info.forward_disc(fake, Fy)

            state_info.optim_Disc.zero_grad()
            loss_real = criterion_GAN(Rout, valid)
            loss_fake = criterion_GAN(Fout, unvalid)
            loss_Disc = (loss_real + loss_fake) / 2
            loss_Disc.backward()
            state_info.optim_Disc.step()

            resultR = label_Ry.eq(Ry).cpu().type(torch.ByteTensor)
            predR = torch.round(Rout).cpu().type(torch.ByteTensor)

            resultF = label_Fy.eq(Fy).cpu().type(torch.ByteTensor)
            predF = torch.round(Fout).cpu().type(torch.ByteTensor)
            
            correctR += float(predR.eq(resultR.data).cpu().sum())
            correctF += float(predR.eq(resultR.data).cpu().sum())

            total += float(real.size(0))

            if it % 10 == 0:
                utils.print_log('Disc Train, {}, {}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'
                      .format(epoch, it, loss_real.item(), loss_fake.item(), 100.*correctR / total, 100.*correctF / total))
                print('Disc Train, {}, {}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'
                      .format(epoch, it, loss_real.item(), loss_fake.item(), 100.*correctR / total, 100.*correctF / total))

        total = torch.tensor(0, dtype=torch.float32)
        # test
        state_info.disc.eval()
        for it, (Noise, Ny, label_Ny) in enumerate(Noise_Test_loader):

            Noise, Ny, label_Ny = to_var(Noise, FloatTensor), to_var(Ny, LongTensor), to_var(label_Ny, LongTensor)

            Nout = state_info.forward_disc(Noise, Ny)

            resultN = label_Ny.eq(Ny).cpu().type(torch.ByteTensor)
            predN = torch.round(Nout).cpu().type(torch.ByteTensor)
            
            correctN += float(predN.eq(resultN.data).cpu().sum())
            total += float(Noise.size(0))

        utils.print_log('Disc Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correctN / total))
        print('Disc Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correctN / total))

        state_info.lr_Disc.step()

    utils.print_log('')

def train_Noise(state_info, Noise_loader, Test_loader, start_epoch, last_epoch): # all 
    
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    criterion = torch.nn.CrossEntropyLoss()

    # criterion_GAN = torch.nn.MSELoss()

    utils.print_log('Type, Epoch, Batch, loss, BCE, KLD, CE')
    

    for epoch in range(start_epoch, last_epoch):

        correct_Noise = torch.tensor(0, dtype=torch.float32)
        correct_Real = torch.tensor(0, dtype=torch.float32)
        correct_Test = torch.tensor(0, dtype=torch.float32)
        total = torch.tensor(0, dtype=torch.float32)

        # train
        state_info.noise.train()
        for it, (Noise, Ny, label_Ny) in enumerate(Noise_loader):

            Noise, Ny, label_Ny = to_var(Noise, FloatTensor), to_var(Ny, LongTensor), to_var(label_Ny, LongTensor)

            Nout = state_info.forward_disc(Noise, Ny)
            normed_Nout = 2 * Nout - 1. # -1 < Nout < 1 

            cls_out = state_info.forward_Noise(Noise, normed_Nout)

            state_info.optim_Noise.zero_grad()
            loss = criterion(cls_out, Ny)
            loss.backward()
            state_info.optim_Noise.step()

            _, pred = torch.max(cls_out.data, 1)
            correct_Noise += float(pred.eq(Ny.data).cpu().sum())
            correct_Real += float(pred.eq(label_Ny.data).cpu().sum())
            total += float(Noise.size(0))
            
            if it % 10 == 0:
                utils.print_log('main Train, {}, {}, {:.6f}, {:.3f}, {:.3f}'
                      .format(epoch, it, loss.item(), 100.*correct_Noise / total, 100.*correct_Real / total))
                print('main Train, {}, {}, {:.6f}, {:.3f}, {:.3f}'
                      .format(epoch, it, loss.item(), 100.*correct_Noise / total, 100.*correct_Real / total))

        total = torch.tensor(0, dtype=torch.float32)
        # test
        state_info.noise.eval()
        for it, (Test, Ty, label_Ty) in enumerate(Test_loader):

            Test, Ty, label_Ty = to_var(Test, FloatTensor), to_var(Ty, LongTensor), to_var(label_Ty, LongTensor)

            cls_out = state_info.forward_Noise(Test, Ty)

            _, pred = torch.max(cls_out.data, 1)
            correct_Test += float(pred.eq(Ty.data).cpu().sum())
            total += float(Noise.size(0))

        utils.print_log('main Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correct_Test / total))
        print('main Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correct_Test / total))

        state_info.lr_Noise.step()


    utils.print_log('')

def train_Base(state_info, All_loader, Test_loader, start_epoch, last_epoch): # all 
    
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    criterion = torch.nn.CrossEntropyLoss()

    # criterion_GAN = torch.nn.MSELoss()

    utils.print_log('Type, Epoch, Batch, loss, BCE, KLD, CE')
    state_info.set_train_mode()

    for epoch in range(start_epoch, last_epoch):

        correct_Noise = torch.tensor(0, dtype=torch.float32)
        correct_Real = torch.tensor(0, dtype=torch.float32)
        correct_Test = torch.tensor(0, dtype=torch.float32)
        total = torch.tensor(0, dtype=torch.float32)

        # train
        state_info.base.train()
        for it, (All, Ay, label_Ay) in enumerate(Noise_loader):

            All, Ay, label_Ay = to_var(All, FloatTensor), to_var(Ay, LongTensor), to_var(label_Ay, LongTensor)

            cls_out = state_info.forward_Base(All)

            state_info.optim_Base.zero_grad()
            loss = criterion(cls_out, Ay)
            loss.backward()
            state_info.optim_Base.step()

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
        state_info.base.eval()
        for it, (Test, Ty, label_Ty) in enumerate(Test_loader):

            Test, Ty, label_Ty = to_var(Test, FloatTensor), to_var(Ty, LongTensor), to_var(label_Ty, LongTensor)

            cls_out = state_info.forward_Base(Test)

            _, pred = torch.max(cls_out.data, 1)
            correct_Test += float(pred.eq(Ty.data).cpu().sum())
            total += float(Noise.size(0))

        utils.print_log('main Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correct_Test / total))
        print('main Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correct_Test / total))

        state_info.lr_Base.step()

    utils.print_log('')
