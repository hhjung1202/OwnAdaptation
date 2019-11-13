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

def train_Base(args, state_info, All_loader, Test_loader): # all 
    
    best_prec_result = torch.tensor(0, dtype=torch.float32)
    start_time = time.time()
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    mode = 'base'
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

        if 100.*correct_Test / total > best_prec_result:
            best_prec_result = 100.*correct_Test / total
            filename = 'checkpoint_best.pth.tar'
            utils.save_state_checkpoint(state_info, best_prec_result, epoch, mode, filename, utils.default_model_dir)

        filename = 'latest.pth.tar'
        utils.save_state_checkpoint(state_info, best_prec_result, epoch, mode, filename, utils.default_model_dir)
        state_info.lr_Base.step()
        utils.print_log('')

    now = time.gmtime(time.time() - start_time)
    utils.print_log('Best Prec : {:.4f}'.format(best_prec_result.item()))
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))

# adversarial_loss = torch.nn.BCELoss()
# criterion_GAN = torch.nn.MSELoss()
# criterion_cycle = torch.nn.L1Loss()
# criterion_identity = torch.nn.L1Loss()
# criterion = nn.CrossEntropyLoss().cuda()