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

def loss_fn(recon_x, x, means, log_var, cls_output, y):
    
    criterion_BCE = torch.nn.BCELoss(reduction='sum')
    criterion = nn.CrossEntropyLoss(reduction='sum')

    BCE = criterion_BCE(recon_x.view(x.size(0), -1), x.view(x.size(0), -1))
    KLD = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp())
    CE = criterion(cls_output, y)

    return (BCE + KLD + CE) / x.size(0), BCE, KLD, CE

def pretrain(args, state_info, train_loader, test_loader, Src_sample):

    start_epoch = 0
    best_prec_result = torch.tensor(0, dtype=torch.float32)

    final_checkpoint = utils.load_checkpoint(utils.default_model_dir, is_final=True, is_source=True)
    if final_checkpoint:
        best_prec_result = checkpoint['Best_Prec']
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

    for epoch in range(start_epoch, args.epoch):
        
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

        batch_size = x.size(0)
        x, y = to_var(x, torch.cuda.FloatTensor), to_var(y, torch.cuda.LongTensor)
        recon_x, means, log_var, z, cls_output = state_info.pretrain_forward(x)

        #  Train 
        state_info.optim_VAE_src.zero_grad()
        loss, BCE, KLD, CE = loss_fn(recon_x, x, means, log_var, cls_output, y)
        loss.backward()
        state_info.optim_VAE_src.step()

        #  Log Print
        total += float(cls_output.size(0))
        _, predicted = torch.max(cls_output.data, 1)
        correct += float(predicted.eq(y.data).cpu().sum())
        
        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}'
                  .format(epoch, it, loss.item(), BCE.item(), KLD.item(), CE.item(), 100.*correct / total))

            print('Train, {}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}'
                  .format(epoch, it, loss.item(), BCE.item(), KLD.item(), CE.item(), 100.*correct / total))

    utils.print_log('')

def test(args, state_info, test_loader, Src_sample, epoch):

    utils.print_log('Type, Epoch, Batch, Acc')
    state_info.pretrain_set_test_mode()
    correct = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)

    for it, (x, y) in enumerate(test_loader):

        batch_size = x.size(0)
        x, y = to_var(x, torch.cuda.FloatTensor), to_var(y, torch.cuda.LongTensor)
        _, cls_output, _ = state_info.pretrain_forward(x, test=True)

        #  Log Print
        total += float(cls_output.size(0))
        _, predicted = torch.max(cls_output.data, 1)
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

    recon_x, _, _ = state_info.pretrain_forward(Src_sample, test=True)
    recon_x = to_data(recon_x)

    concat = merge_images(Src_sample, recon_x)

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
