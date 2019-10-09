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

# lr, batch

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

def train_Disc(args, state_info, True_loader, Fake_loader, Noise_Test_loader): # all 

    best_prec_result = torch.tensor(0, dtype=torch.float32)
    start_time = time.time()
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    mode = 'disc'
    utils.default_model_dir = os.path.join(args.dir, mode)

    criterion_GAN = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    # criterion_GAN = torch.nn.MSELoss()

    percentage = get_percentage_Fake(Fake_loader)

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

    for epoch in range(start_epoch, args.d_epoch):

        correctR = torch.tensor(0, dtype=torch.float32)
        correctF = torch.tensor(0, dtype=torch.float32)
        correctN = torch.tensor(0, dtype=torch.float32)
        total = torch.tensor(0, dtype=torch.float32)

        # train
        state_info.disc.train()
        for it, ((real, Ry, label_Ry), (fake, Fy, label_Fy)) in enumerate(zip(True_loader, Fake_loader)):

            # valid = Variable(FloatTensor(real.size(0), 1).fill_(1.0), requires_grad=False)
            # unvalid = Variable(FloatTensor(fake.size(0), 1).fill_(0.0), requires_grad=False)
            valid = Variable(LongTensor(real.size(0), 1).fill_(1), requires_grad=False)
            unvalid = Variable(LongTensor(fake.size(0), 1).fill_(0), requires_grad=False)

            real, Ry, label_Ry = to_var(real, FloatTensor), to_var(Ry, LongTensor), to_var(label_Ry, LongTensor)
            fake, Fy, label_Fy = to_var(fake, FloatTensor), to_var(Fy, LongTensor), to_var(label_Fy, LongTensor)

            Rout, Fout = state_info.forward_disc(real, Ry), state_info.forward_disc(fake, Fy)

            state_info.optim_Disc.zero_grad()
            # loss_real = criterion_GAN(Rout, valid)
            # loss_fake = criterion_GAN(Fout, unvalid)
            loss_real = criterion(Rout, valid)
            loss_fake = criterion(Fout, unvalid)
            loss_Disc = (loss_real + loss_fake) / 2
            loss_Disc.backward()
            state_info.optim_Disc.step()

            _, predR = torch.max(Rout.data, 1)
            resultR = label_Ry.eq(Ry).cpu().type(torch.ByteTensor).view(-1,1)

            _, predF = torch.max(Fout.data, 1)
            resultF = label_Fy.eq(Fy).cpu().type(torch.ByteTensor).view(-1,1)
            
            correctR += float(predR.eq(resultR.data).cpu().sum())
            correctF += float(predF.eq(resultF.data).cpu().sum())


            # resultR = label_Ry.eq(Ry).cpu().type(torch.ByteTensor).view(-1,1)
            # predR = torch.round(Rout).cpu().type(torch.ByteTensor)

            # resultF = label_Fy.eq(Fy).cpu().type(torch.ByteTensor).view(-1,1)
            # predF = torch.round(Fout).cpu().type(torch.ByteTensor)
            
            # correctR += float(predR.eq(resultR.data).cpu().sum())
            # correctF += float(predF.eq(resultF.data).cpu().sum())

            total += float(real.size(0))

            if it % 10 == 0:

                utils.print_log('Disc Train, {}, {}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'
                      .format(epoch, it, loss_real.item(), loss_fake.item(), 100.*correctR / total, 100.*correctF / total))
                print('Disc Train, {}, {}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'
                      .format(epoch, it, loss_real.item(), loss_fake.item(), 100.*correctR / total, 100.*correctF / total))


        # test
        state_info.disc.eval()
        total = torch.tensor(0, dtype=torch.float32)
        for it, (Noise, Ny, label_Ny) in enumerate(Noise_Test_loader):

            Noise, Ny, label_Ny = to_var(Noise, FloatTensor), to_var(Ny, LongTensor), to_var(label_Ny, LongTensor)

            Nout = state_info.forward_disc(Noise, Ny)

            resultN = label_Ny.eq(Ny).cpu().type(torch.ByteTensor).view(-1,1)
            _, predN = torch.max(Nout.data, 1)

            # resultN = label_Ny.eq(Ny).cpu().type(torch.ByteTensor).view(-1,1)
            # predN = torch.round(Nout).cpu().type(torch.ByteTensor)
            
            correctN += float(predN.eq(resultN.data).cpu().sum())
            total += float(Noise.size(0))

        utils.print_log('Disc Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correctN / total))
        print('Disc Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correctN / total))

        if 100.*correctN / total > best_prec_result:
            best_prec_result = 100.*correctN / total
            filename = 'checkpoint_best.pth.tar'
            utils.save_state_checkpoint(state_info, best_prec_result, epoch, mode, filename, utils.default_model_dir)

        filename = 'latest.pth.tar'
        utils.save_state_checkpoint(state_info, best_prec_result, epoch, mode, filename, utils.default_model_dir)
        state_info.lr_Disc.step()
        utils.print_log('')

    now = time.gmtime(time.time() - start_time)
    utils.print_log('Best Prec : {:.4f}'.format(best_prec_result.item()))
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))



# adversarial_loss = torch.nn.BCELoss()
# criterion_GAN = torch.nn.MSELoss()
# criterion_cycle = torch.nn.L1Loss()
# criterion_identity = torch.nn.L1Loss()
# criterion = nn.CrossEntropyLoss().cuda()