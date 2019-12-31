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

    def model_init(self, Img=[1, 28], H=400, latent_size=20, num_class=10):
        self.AE = AE(img_D=get_size(Img), H=H, latent_size=latent_size, num_class=num_class)
        self.len = Img[1]

    def forward(self, x, test=False):
        x_hat, z = self.AE(x)
        x_hat = x_hat.view(x_hat.size(0), -1, self.len, self.len)

        return x_hat, z

    def model_cuda_init(self):
        if torch.cuda.is_available():
            self.AE = nn.DataParallel(self.AE).cuda()

    def weight_init(self):
        self.AE.apply(self.weights_init_normal)

    def weights_init_normal(self, m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def optimizer_init(self, args):
        self.optim_AE = optim.Adam(self.AE.parameters(), lr=args.lr)

    def learning_scheduler_init(self, args, load_epoch=0):
        self.lr_AE = optim.lr_scheduler.StepLR(self.optim_AE, step_size=30, gamma=0.1)

    def learning_step(self):
        self.lr_AE.step()

    def set_train_mode(self):
        self.AE.train()

    def set_test_mode(self):
        self.AE.eval()

    def load_state_dict(self, checkpoint):
        self.AE.load_state_dict(checkpoint['AE_dict'])
        self.optim_AE.load_state_dict(checkpoint['AE_optimizer'])

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def print_log(text, filename="log.csv"):
    if not os.path.exists(default_model_dir):
        os.makedirs(default_model_dir)
    model_filename = os.path.join(default_model_dir, filename)
    with open(model_filename, "a") as myfile:
        myfile.write(text + "\n")