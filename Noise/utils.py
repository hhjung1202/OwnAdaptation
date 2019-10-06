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

    def model_init(self, args):
        self.base = Basic_Classifier(chIn=args.chIn, clsN=args.clsN, resnet_layer=args.layer) # input : [z, y] def __init__(self, chIn=1, clsN=10, resnet_layer=20):
        self.disc = Discriminator(chIn=args.chIn, clsN=args.clsN) # input : [z, y] def __init__(self, chIn=1, clsN=10):
        self.noise = Classifier(chIn=args.chIn, clsN=args.clsN, resnet_layer=args.layer) # input : [z, y] def __init__(self, chIn=1, clsN=10, resnet_layer=20):
        
    def forward_disc(self, image, label):
        label_one = torch.cuda.FloatTensor(image.size(0), 10).zero_().scatter_(1, label.view(-1, 1), 1)
        out = self.disc(image, label_one)
        return out

    def forward_Noise(self, image, gamma):
        out = self.noise(image, gamma)
        return out

    def forward_Base(self, image):
        out = self.base(image)
        return out

    def model_cuda_init(self):
        if torch.cuda.is_available():
            self.base = nn.DataParallel(self.base).cuda()
            self.disc = nn.DataParallel(self.disc).cuda()
            self.noise = nn.DataParallel(self.noise).cuda()

    def weight_cuda_init(self):
        self.base.apply(self.weights_init_normal)
        self.disc.apply(self.weights_init_normal)
        self.noise.apply(self.weights_init_normal)
        
    def weights_init_normal(self, m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def optimizer_init(self, args):
        self.optim_Base = optim.Adam(self.base.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay) # lr, b1, b2, weight_decay
        self.optim_Disc = optim.Adam(self.disc.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay)
        self.optim_Noise = optim.Adam(self.noise.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay)

        # self.optim_Base = optim.SGD(self.base.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # self.optim_Disc = optim.SGD(self.disc.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # self.optim_Noise = optim.SGD(self.noise.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    def learning_scheduler_init(self, args):
        # self.lr_Base = optim.lr_scheduler.LambdaLR(self.optimizer_G_AB, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        # self.lr_Disc = optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        # self.lr_Noise = optim.lr_scheduler.LambdaLR(self.optimizer_G_BA, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        
        self.lr_Base = optim.lr_scheduler.MultiStepLR(self.optim_Base, args.milestones, gamma=args.gamma, last_epoch=args.last_epoch)
        self.lr_Disc = optim.lr_scheduler.MultiStepLR(self.optim_Disc, args.Dmilestones, gamma=args.gamma, last_epoch=args.last_epoch)
        self.lr_Noise = optim.lr_scheduler.MultiStepLR(self.optim_Noise, args.milestones, gamma=args.gamma, last_epoch=args.last_epoch)

    def load_state_dict(self, checkpoint, mode):
        if mode == "disc":
            self.disc.load_state_dict(checkpoint['weight'])
            self.optim_Disc.load_state_dict(checkpoint['optim'])
        elif mode == "noise":
            self.noise.load_state_dict(checkpoint['weight'])
            self.optim_Noise.load_state_dict(checkpoint['optim'])   
        elif mode == "base":
            self.base.load_state_dict(checkpoint['weight'])
            self.optim_Base.load_state_dict(checkpoint['optim'])

        

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

def save_state_checkpoint(state_info, best_prec_result, epoch, mode, filename, directory):
    if mode == "disc":
        model = state_info.disc,
        weight = state_info.disc.state_dict()
        optim = state_info.optim_Disc.state_dict(),
    elif mode == "noise":
        model = state_info.noise,
        weight = state_info.noise.state_dict()
        optim = state_info.optim_Noise.state_dict(),
    elif mode == "base":
        model = state_info.base,
        weight = state_info.base.state_dict()
        optim = state_info.optim_Base.state_dict(),

    print(state_info.optim_Noise.state_dict()['param_groups'])
    print(optim['param_groups'])

    save_checkpoint({
        'epoch': epoch,
        'Best_Prec': best_prec_result,
        'model': model,
        'weight': weight,
        'optim': optim,
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
