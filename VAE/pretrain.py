import argparse
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from model import *
import os
import torch.backends.cudnn as cudnn
import time
import utils
import dataset
import math


# FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def loss_function(args, x_hat, x, mu, log_var, cls, y):

    criterion_BCE = torch.nn.BCELoss(reduction='sum')
    criterion = torch.nn.CrossEntropyLoss()

    BCE = criterion_BCE(x_hat, x.view(x.size(0), -1))
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    if y.size(-1) == 1:
        y=y.view(-1)

    CE = args.CE * criterion(cls, y)
    return BCE + KLD + CE, BCE.item(), KLD.item(), CE.item()

def pretrain(args, state_info, train_loader, test_loader, Src_sample):

    start_epoch = 0
    best_prec_result = torch.tensor(0, dtype=torch.float32)

    final_checkpoint = utils.load_checkpoint(utils.default_model_dir, is_final=True, is_source=True)
    if final_checkpoint:
        best_prec_result = final_checkpoint['Best_Prec']
        state_info.pretrain_load_state_dict(final_checkpoint)
        print('load pretrained final model, best_prec_result :', best_prec_result)
        return

    checkpoint = utils.load_checkpoint(utils.default_model_dir, is_last=True, is_source=True)    
    if not checkpoint:
        state_info.pretrain_learning_scheduler_init(args)
    else:
        start_epoch = checkpoint['epoch'] + 1
        best_prec_result = checkpoint['Best_Prec']
        state_info.pretrain_load_state_dict(checkpoint)
        state_info.pretrain_learning_scheduler_init(args, load_epoch=start_epoch)

    for epoch in range(start_epoch, args.Pepoch):
        
        train(args, state_info, train_loader, epoch)
        prec_result = test(args, state_info, test_loader, Src_sample, epoch)

        if prec_result > best_prec_result:
            best_prec_result = prec_result
            filename = 'source_checkpoint_best.pth.tar'
            utils.save_source_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)

        filename = 'source_latest.pth.tar'
        utils.save_source_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)
        state_info.pretrain_learning_step()

    filename = 'source_final.pth.tar'
    utils.save_source_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)


def train(args, state_info, train_loader, epoch): # all 

    utils.print_log('Type, Epoch, Batch, loss, BCE, KLD, CE, Acc')
    state_info.pretrain_set_train_mode()
    correct = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)

    for it, (x, y) in enumerate(train_loader):

        x, y = to_var(x, torch.cuda.FloatTensor), to_var(y, torch.cuda.LongTensor)
        x_hat, mu, log_var, z, cls = state_info.pretrain_forward(x)

        #  Train 
        state_info.optim_VAE_src.zero_grad()
        loss, BCE, KLD, CE = loss_function(args, x_hat, x, mu, log_var, cls, y)
        loss.backward(retain_graph=True)
        state_info.optim_VAE_src.step()

        #  Log Print
        total += float(cls.size(0))
        _, predicted = torch.max(cls.data, 1)
        correct += float(predicted.eq(y.data).cpu().sum())
        
        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.2f}'
                  .format(epoch, it, loss.item(), BCE, KLD, CE, 100.*correct / total))

            print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.2f}'
                  .format(epoch, it, loss.item(), BCE, KLD, CE, 100.*correct / total))

    utils.print_log('')

def test(args, state_info, test_loader, Src_sample, epoch):

    utils.print_log('Type, Epoch, Batch, Acc')
    state_info.pretrain_set_test_mode()
    correct = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)

    for it, (x, y) in enumerate(test_loader):

        x, y = to_var(x, torch.cuda.FloatTensor), to_var(y, torch.cuda.LongTensor)
        _, cls, _ = state_info.pretrain_forward(x, test=True)

        #  Log Print
        total += float(cls.size(0))
        _, predicted = torch.max(cls.data, 1)
        correct += float(predicted.eq(y.data).cpu().sum())
        
        if it % 10 == 0:
            utils.print_log('Test, {}, {}, {:.2f}'.format(epoch, it, 100.*correct / total))
            print('Test, {}, {}, {:.2f}'.format(epoch, it, 100.*correct / total))
    
    make_sample_image(state_info, Src_sample, epoch)
    utils.print_log('')

    return 100.*correct / total

def make_sample_image(state_info, Src_sample, epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    img_path = utils.make_directory(os.path.join(utils.default_model_dir, 'images/Pretrain'))

    x_hat, _, _ = state_info.pretrain_forward(Src_sample, test=True)
    Src_sample, x_hat = to_data(Src_sample), to_data(x_hat)

    concat = merge_images(Src_sample, x_hat)

    save_image(concat.data, os.path.join(img_path, '%d.png' % epoch), normalize=True)

def merge_images(sources, targets, row=10):
    _, _, h, w = sources.shape
    merged = np.zeros([3, row*h, row*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        if i is row:
            break
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t

    return torch.from_numpy(merged)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x, dtype):
    return Variable(x.type(dtype))
