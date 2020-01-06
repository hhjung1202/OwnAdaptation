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
from .trainFunction import *


'''
    2가지 과정이 필요해보인다
    1. 초기 방식
    2. Pseudo label을 가지고 모든것 학습(random label 사용?)
    3. 최종 Pseudo label 어떻게 사용할지 결정??
'''

def train_NAE(args, state_info, Train_loader, Test_loader): # all 

    Memory = MemorySet(args=args)
    
    start_time = time.time()
    best_prec_result = torch.tensor(0, dtype=torch.float32)
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    mode = args.model
    utils.default_model_dir = os.path.join(args.dir, mode)
    
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    start_epoch = 0
    checkpoint = None
    # checkpoint = utils.load_checkpoint(utils.default_model_dir)    
    if not checkpoint:
        args.last_epoch = -1
        state_info.learning_scheduler_init(args, mode)
    else:
        start_epoch = checkpoint['epoch'] + 1
        best_prec_result = checkpoint['Best_Prec']
        state_info.load_state_dict(checkpoint, mode)
        args.last_epoch = start_epoch
        state_info.learning_scheduler_init(args, mode)

    for epoch in range(0, args.epoch1):

        epoch_result = train_step1(state_info, Train_loader, Test_loader, Memory, criterion, epoch)
        if epoch_result > best_prec_result:
            best_prec_result = epoch_result
        state_info.lr_model.step()
        utils.print_log('')

    for epoch in range(0, args.epoch2):

        epoch_result = train_step2(args, state_info, Train_loader, Test_loader, Memory, criterion, epoch)
        if epoch_result > best_prec_result:
            best_prec_result = epoch_result
        state_info.lr_model.step()
        utils.print_log('')

    pseudo_loader = get_Pseudo_loader(args, state_info, Memory)

    for epoch in range(0, args.epoch3):

        epoch_result = train_step3(args, state_info, pseudo_loader, Test_loader, Memory, criterion, epoch)
        if epoch_result > best_prec_result:
            best_prec_result = epoch_result
        state_info.lr_model.step()
        utils.print_log('')

    pseudo_loader = get_Pseudo_loader(args, state_info, Memory)

    for epoch in range(0, args.epoch4):

        epoch_result = train_step3(args, state_info, pseudo_loader, Test_loader, Memory, criterion, epoch)
        if epoch_result > best_prec_result:
            best_prec_result = epoch_result
        state_info.lr_model.step()
        utils.print_log('')

    now = time.gmtime(time.time() - start_time)
    utils.print_log('Best Prec : {:.4f}'.format(best_prec_result.item()))
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))

# adversarial_loss = torch.nn.BCELoss()
# criterion_GAN = torch.nn.MSELoss()
# criterion_cycle = torch.nn.L1Loss()
# criterion_identity = torch.nn.L1Loss()
# criterion = nn.CrossEntropyLoss().cuda()

    # for epoch in range(start_epoch, args.epoch):

    #     epoch_result = train_step3(state_info, Train_loader, Test_loader, Memory, criterion, epoch)

    #     if epoch_result > best_prec_result:
    #         best_prec_result = epoch_result
    #         filename = 'checkpoint_best.pth.tar'
    #         utils.save_state_checkpoint(state_info, best_prec_result, epoch, mode, filename, utils.default_model_dir)

    #     filename = 'latest.pth.tar'
    #     utils.save_state_checkpoint(state_info, best_prec_result, epoch, mode, filename, utils.default_model_dir)
    #     state_info.lr_model.step()
    #     utils.print_log('')