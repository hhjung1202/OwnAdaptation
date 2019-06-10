import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
from model import *
import csv
import random
import os
import math
import numbers

default_model_dir = "./"
c = None
str_w = ''
csv_file_name = 'weight.csv'

def get_size(img=[1,28]):
    return img[0]*img[1]*img[1]

class model_optim_state_info(object):
    def __init__(self):
        pass

    def model_init(self, Img_S=[1, 28], Img_T=[1, 28], H=400, latent_size=20, num_class=10):
        self.VAE_src = VAE(img_D=get_size(Img_S), H=H, latent_size=latent_size, num_class=num_class)
        self.VAE_tgt = VAE(img_D=get_size(Img_T), H=H, latent_size=latent_size, num_class=num_class)
        self.lenS = Img_S[1]
        self.lenT = Img_T[1]

    def pretrain_forward(self, x, test=False):
        x_hat, mu, log_var, z, cls = self.VAE_src(x)
        x_hat = x_hat.view(recover.size(0), -1, self.lenS, self.lenS)

        if not test:
            return x_hat, mu, log_var, z, cls
        else:
            return x_hat, cls, z

    def forward(self, x, test=False):
        x_hat_t, mu, log_var, z, clsT = self.VAE_tgt(x)
        x_hat_s, clsS = self.VAE_src(None, z=z)

        x_hat_t = x_hat_t.view(x_hat_t.size(0), -1, self.lenT, self.lenT)
        x_hat_s = x_hat_s.view(x_hat_s.size(0), -1, self.lenS, self.lenS)

        if not test:
            return x_hat_t, mu, log_var, z, clsT, clsS.detach()
        else:
            return clsT, clsS, x_hat_t, x_hat_s # edit

    def forward_z(self, z):
        x_hat_s, clsS = self.VAE_src(None, z=z)
        x_hat_t, clsT = self.VAE_tgt(None, z=z)

        x_hat_t = x_hat_t.view(x_hat_t.size(0), -1, self.lenT, self.lenT)
        x_hat_s = x_hat_s.view(x_hat_s.size(0), -1, self.lenS, self.lenS)
        
        return x_hat_s, x_hat_t

    def model_cuda_init(self):
        if torch.cuda.is_available():
            self.VAE_src = nn.DataParallel(self.VAE_src).cuda()
            self.VAE_tgt = nn.DataParallel(self.VAE_tgt).cuda()

    def weight_init(self):
        self.VAE_src.apply(self.weights_init_normal)
        self.VAE_tgt.apply(self.weights_init_normal)

    def weights_init_normal(self, m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def optimizer_init(self, args):
        self.optim_VAE_src = optim.Adam(self.VAE_src.parameters(), lr=args.lr)
        self.optim_VAE_tgt = optim.Adam(self.VAE_tgt.parameters(), lr=args.lr) #, b1=args.b1, b2=args.b2, weight_decay=args.weight_decay

    def learning_scheduler_init(self, args, load_epoch=0):
        # self.lr_VAE_tgt = optim.lr_scheduler.LambdaLR(self.optim_VAE_tgt, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        self.lr_VAE_tgt = optim.lr_scheduler.StepLR(self.optim_VAE_tgt, step_size=30, gamma=0.1)
        
    def learning_step(self):
        self.lr_VAE_tgt.step()

    def set_train_mode(self):
        self.VAE_tgt.train()

    def set_test_mode(self):
        self.VAE_tgt.eval()

    def load_state_dict(self, checkpoint):
        self.VAE_tgt.load_state_dict(checkpoint['VAE_tgt_dict'])
        self.optim_VAE_tgt.load_state_dict(checkpoint['VAE_tgt_optimizer'])

    def pretrain_learning_scheduler_init(self, args, load_epoch=0):
        # self.lr_VAE_src = optim.lr_scheduler.LambdaLR(self.optim_VAE_src, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step) # NEED TO EDIT!!!!!!
        self.lr_VAE_src = optim.lr_scheduler.StepLR(self.optim_VAE_src, step_size=30, gamma=0.1)

    def pretrain_learning_step(self):
        self.lr_VAE_src.step()

    def pretrain_set_train_mode(self):
        self.VAE_src.train()

    def pretrain_set_test_mode(self):
        self.VAE_src.eval()

    def pretrain_load_state_dict(self, checkpoint):
        self.VAE_src.load_state_dict(checkpoint['VAE_src_dict'])
        self.optim_VAE_src.load_state_dict(checkpoint['VAE_src_optimizer'])

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def save_source_checkpoint(state_info, best_prec_result, filename, directory, epoch):
    save_checkpoint({
        'epoch': epoch,
        'Best_Prec': best_prec_result,

        'VAE_src_model': state_info.VAE_src,
        'VAE_src_dict': state_info.VAE_src.state_dict(),
        'VAE_src_optimizer': state_info.optim_VAE_src.state_dict(),

    }, filename, directory)

def save_target_checkpoint(state_info, best_prec_result, filename, directory, epoch):
    save_checkpoint({
        'epoch': epoch,
        'Best_Prec': best_prec_result,

        'VAE_tgt_model': state_info.VAE_tgt,
        'VAE_tgt_dict': state_info.VAE_tgt.state_dict(),
        'VAE_tgt_optimizer': state_info.optim_VAE_tgt.state_dict(),

    }, filename, directory)


def save_checkpoint(state, filename, model_dir):

    # model_dir = 'drive/app/torch/save_Routing_Gate_2'
    model_filename = os.path.join(model_dir, filename)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(state, model_filename)
    print("=> saving checkpoint '{}'".format(model_filename))

    return

def load_checkpoint(directory, is_last=True, is_source=True, target_num=10, is_final=False):

    if is_final:
        if is_source:
            load_state_name = os.path.join(directory, 'source_final.pth.tar')
        else:
            load_state_name = os.path.join(directory, 'target_final.pth.tar')

    else:
        if is_last:
            if is_source:
                load_state_name = os.path.join(directory, 'source_latest.pth.tar')
            else:
                load_state_name = os.path.join(directory, 'target_latest.pth.tar')
        else:
            if is_source:
                load_state_name = os.path.join(directory, 'source_checkpoint_best.pth.tar')
            else:
                load_state_name = os.path.join(directory, 'target_checkpoint_{}.pth.tar'.format(target_num))
    
    if os.path.exists(load_state_name):
        print("=> loading checkpoint '{}'".format(load_state_name))
        state = torch.load(load_state_name)
        return state
    else:
        return None

def print_log(text, filename="log.csv"):
    if not os.path.exists(default_model_dir):
        os.makedirs(default_model_dir)
    model_filename = os.path.join(default_model_dir, filename)
    with open(model_filename, "a") as myfile:
        myfile.write(text + "\n")

def extract_mapping(x, y):
    if not os.path.exists(default_model_dir):
        os.makedirs(default_model_dir)
    
    model_filename = os.path.join(default_model_dir, csv_file_name)
    for _, ((x_), (y_)) in enumerate(zip(x, y)):
        with open(model_filename, 'a') as fout:
            fout.write(x_ + "," + y_ + "\n")