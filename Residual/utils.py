import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from model import *
import csv
import random
import os

default_model_dir = "./"
c = None
str_w = ''
csv_file_name = 'weight.csv'


class model_optim_state_info(object):
    def __init__(self):
        pass

    def model_init(self):
        self.G_Residual = Generator_Residual(tgt_ch=3, src_ch=1, out_ch=3) # input : [z, y]
        self.D_tgt = Discriminator(input_ch=3) # input : x_src, G_AB
        
        self.G_Restore = Generator_Restore(fake_ch=3, base_ch=3, out_ch=1) # input : [z, y]
        self.D_src = Discriminator(input_ch=1) # input : x_target, G_BA

        self.cls_src = Classifier(input_ch=1) # input: G_AB
        self.cls_target = Classifier(input_ch=3) # input: G_BA

    def forward(self, real_S, shuffle_T):
        fake_T = self.G_Residual(real_S, shuffle_T)
        fake_S = self.G_Restore(fake_T, shuffle_T)
        output_cls_recov = self.cls_src(fake_S)
        output_cls_target = self.cls_target(fake_T)
        return fake_T, fake_S, output_cls_recov, output_cls_target
        
    def model_cuda_init(self):
        if torch.cuda.is_available():
            self.G_Residual = nn.DataParallel(self.G_Residual).cuda()
            self.D_tgt = nn.DataParallel(self.D_tgt).cuda()
            self.G_Restore = nn.DataParallel(self.G_Restore).cuda()
            self.D_src = nn.DataParallel(self.D_src).cuda()
            self.cls_src = nn.DataParallel(self.cls_src).cuda()
            self.cls_target = nn.DataParallel(self.cls_target).cuda()

    def weight_cuda_init(self):
        self.G_Residual.apply(self.weights_init_normal)
        self.D_tgt.apply(self.weights_init_normal)
        self.G_Restore.apply(self.weights_init_normal)
        self.D_src.apply(self.weights_init_normal)
        self.cls_src.apply(self.weights_init_normal)
        self.cls_target.apply(self.weights_init_normal)

    def weights_init_normal(self, m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def optimizer_init(self, lr, b1, b2, weight_decay):
        self.optim_G_Residual = optim.Adam(self.G_Residual.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optim_D_tgt = optim.Adam(self.D_tgt.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optim_G_Restore = optim.Adam(self.G_Restore.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optim_D_src = optim.Adam(self.D_src.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optim_CS = optim.Adam(self.cls_src.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optim_CT = optim.Adam(self.cls_target.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)

    def learning_scheduler_init(self, args, load_epoch=0):
        self.lr_G_Residual = optim.lr_scheduler.LambdaLR(self.optim_G_Residual, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        self.lr_D_tgt = optim.lr_scheduler.LambdaLR(self.optim_D_tgt, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        self.lr_G_Restore = optim.lr_scheduler.LambdaLR(self.optim_G_Restore, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        self.lr_D_src = optim.lr_scheduler.LambdaLR(self.optim_D_src, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        self.lr_CS = optim.lr_scheduler.LambdaLR(self.optim_CS, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        self.lr_CT = optim.lr_scheduler.LambdaLR(self.optim_CT, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        
    def learning_step(self):
        self.lr_G_Residual.step()
        self.lr_D_tgt.step()
        self.lr_G_Restore.step()
        self.lr_D_src.step()
        self.lr_CS.step()
        self.lr_CT.step()

    def set_train_mode(self):
        self.G_Residual.train()
        self.D_tgt.train()
        self.G_Restore.train()
        self.D_src.train()
        self.cls_src.train()
        self.cls_target.train()

    def set_test_mode(self):
        self.G_Residual.eval()
        self.D_tgt.eval()
        self.G_Restore.eval()
        self.D_src.eval()
        self.cls_src.eval()
        self.cls_target.eval()

    def load_state_dict(self, checkpoint):
        self.G_Residual.load_state_dict(checkpoint['G_Residual_dict'])
        self.D_tgt.load_state_dict(checkpoint['D_tgt_dict'])
        self.G_Restore.load_state_dict(checkpoint['G_Restore_dict'])
        self.D_src.load_state_dict(checkpoint['D_src_dict'])
        self.cls_src.load_state_dict(checkpoint['CSdict'])
        self.cls_target.load_state_dict(checkpoint['CTdict'])

        self.optim_G_Residual.load_state_dict(checkpoint['G_Residual_optimizer'])
        self.optim_D_tgt.load_state_dict(checkpoint['D_tgt_optimizer'])
        self.optim_G_Restore.load_state_dict(checkpoint['G_Restore_optimizer'])
        self.optim_D_src.load_state_dict(checkpoint['D_src_optimizer'])
        self.optim_CS.load_state_dict(checkpoint['CSoptimizer'])
        self.optim_CT.load_state_dict(checkpoint['CToptimizer'])


class ImagePool():
    def __init__(self, max_size=1024):
        self.max_size = max_size
        self.data = []

    def query(self, data):
        if self.max_size <= 0:  # if the buffer size is 0, do nothing
            return images
        to_return = []
        for image in data.data:
            image = torch.unsqueeze(image, 0)
            if len(self.data) < self.max_size:
                self.data.append(image)
                to_return.append(image)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = image
                else:
                    to_return.append(image)
        return Variable(torch.cat(to_return))

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

def save_state_checkpoint(state_info, best_prec_result, filename, directory, epoch):
    save_checkpoint({
        'epoch': epoch,
        'Best_Prec': best_prec_result,

        'G_Residual_model': state_info.G_Residual,
        'G_Residual_dict': state_info.G_Residual.state_dict(),
        'G_Residual_optimizer': state_info.optim_G_Residual.state_dict(),

        'D_tgt_model': state_info.D_tgt,
        'D_tgt_dict': state_info.D_tgt.state_dict(),
        'D_tgt_optimizer': state_info.optim_D_tgt.state_dict(),

        'G_Restore_model': state_info.G_Restore,
        'G_Restore_dict': state_info.G_Restore.state_dict(),
        'G_Restore_optimizer': state_info.optim_G_Restore.state_dict(),

        'D_src_model': state_info.D_src,
        'D_src_dict': state_info.D_src.state_dict(),
        'D_src_optimizer': state_info.optim_D_src.state_dict(),

        'CSmodel': state_info.cls_src,
        'CSdict': state_info.cls_src.state_dict(),
        'CSoptimizer': state_info.optim_CS.state_dict(),

        'CTmodel': state_info.cls_target,
        'CTdict': state_info.cls_target.state_dict(),
        'CToptimizer': state_info.optimizer_CT.state_dict(),

    }, filename, directory)

def save_checkpoint(state, filename, model_dir):

    # model_dir = 'drive/app/torch/save_Routing_Gate_2'
    model_filename = os.path.join(model_dir, filename)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(state, model_filename)
    print("=> saving checkpoint '{}'".format(model_filename))

    return

def load_checkpoint(directory, is_last=True):

    if is_last:
        load_state_name = os.path.join(directory, 'latest.pth.tar')
    else:
        load_state_name = os.path.join(directory, 'checkpoint_best.pth.tar')
    
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
