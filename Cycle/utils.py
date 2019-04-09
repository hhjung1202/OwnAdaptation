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
        self.G_AB = Generator(in_channels=3, out_channels=3) # input : [z, y]
        self.D_A = Discriminator(in_channels=3) # input : x_src, G_AB
        
        self.G_BA = Generator(in_channels=3, out_channels=1) # input : [z, y]
        self.D_B = Discriminator(in_channels=3) # input : x_target, G_BA

        self.cls_src = Classifier(x_dim=3) # input: G_AB
        self.cls_target = Classifier(x_dim=3) # input: G_BA
        
    def model_cuda_init(self):
        if torch.cuda.is_available():
            self.G_AB = nn.DataParallel(self.G_AB).cuda()
            self.D_A = nn.DataParallel(self.D_A).cuda()
            self.G_BA = nn.DataParallel(self.G_BA).cuda()
            self.D_B = nn.DataParallel(self.D_B).cuda()
            self.cls_src = nn.DataParallel(self.cls_src).cuda()
            self.cls_target = nn.DataParallel(self.cls_target).cuda()

    def weight_cuda_init(self):
        self.G_AB.apply(self.weights_init_normal)
        self.D_A.apply(self.weights_init_normal)
        self.G_BA.apply(self.weights_init_normal)
        self.D_B.apply(self.weights_init_normal)
        self.cls_src.apply(self.weights_init_normal)
        self.cls_target.apply(self.weights_init_normal)

    def weights_init_normal(self, m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def optimizer_init(self, lr, b1, b2, weight_decay):
        self.optimizer_G_AB = optim.Adam(self.G_AB.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_G_BA = optim.Adam(self.G_BA.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_CS = optim.Adam(self.cls_src.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_CT = optim.Adam(self.cls_src.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)

    def learning_scheduler_init(self, args, load_epoch=0):
        self.lr_scheduler_G_AB = optim.lr_scheduler.LambdaLR(self.optimizer_G_AB, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        self.lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        self.lr_scheduler_G_BA = optim.lr_scheduler.LambdaLR(self.optimizer_G_BA, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        self.lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        self.lr_scheduler_CS = optim.lr_scheduler.LambdaLR(self.optimizer_CS, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        self.lr_scheduler_CT = optim.lr_scheduler.LambdaLR(self.optimizer_CT, lr_lambda=LambdaLR(args.epoch, load_epoch, args.decay_epoch).step)
        
    def learning_step(self):
        self.lr_scheduler_G_AB.step()
        self.lr_scheduler_D_A.step()
        self.lr_scheduler_G_BA.step()
        self.lr_scheduler_D_B.step()
        self.lr_scheduler_CS.step()
        self.lr_scheduler_CT.step()

    def set_train_mode(self):
        self.G_AB.train()
        self.D_A.train()
        self.G_BA.train()
        self.D_B.train()
        self.cls_src.train()
        self.cls_target.train()

    def set_test_mode(self):
        self.G_AB.eval()
        self.D_A.eval()
        self.G_BA.eval()
        self.D_B.eval()
        self.cls_src.eval()
        self.cls_target.eval()

    def load_state_dict(self, checkpoint):
        self.G_AB.load_state_dict(checkpoint['GABdict'])
        self.D_A.load_state_dict(checkpoint['DAdict'])
        self.G_BA.load_state_dict(checkpoint['GBAdict'])
        self.D_B.load_state_dict(checkpoint['DBdict'])
        self.cls_src.load_state_dict(checkpoint['CSdict'])
        self.cls_target.load_state_dict(checkpoint['CTdict'])

        self.optimizer_G_AB.load_state_dict(checkpoint['GABoptimizer'])
        self.optimizer_D_A.load_state_dict(checkpoint['DAoptimizer'])
        self.optimizer_G_BA.load_state_dict(checkpoint['GBAoptimizer'])
        self.optimizer_D_B.load_state_dict(checkpoint['DBoptimizer'])
        self.optimizer_CS.load_state_dict(checkpoint['CSoptimizer'])
        self.optimizer_CT.load_state_dict(checkpoint['CToptimizer'])


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

        'GABmodel': state_info.G_AB,
        'GABdict': state_info.G_AB.state_dict(),
        'GABoptimizer': state_info.optimizer_G_AB.state_dict(),

        'DAmodel': state_info.D_A,
        'DAdict': state_info.D_A.state_dict(),
        'DAoptimizer': state_info.optimizer_D_A.state_dict(),

        'GBAmodel': state_info.G_BA,
        'GBAdict': state_info.G_BA.state_dict(),
        'GBAoptimizer': state_info.optimizer_G_BA.state_dict(),

        'DBmodel': state_info.D_B,
        'DBdict': state_info.D_B.state_dict(),
        'DBoptimizer': state_info.optimizer_D_B.state_dict(),

        'CSmodel': state_info.cls_src,
        'CSdict': state_info.cls_src.state_dict(),
        'CSoptimizer': state_info.optimizer_CS.state_dict(),

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
