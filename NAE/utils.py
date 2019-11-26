import torch
import torch.nn as nn
import torch.optim as optim
from model import *
import os

default_model_dir = "./"
c = None
str_w = ''
csv_file_name = 'weight.csv'


class model_optim_state_info(object):
    def __init__(self):
        pass

    def model_init(self, args):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        
        self.init_lr = args.lr
        self.NAE = NAE(I=args.img_size, H=args.h, latent_size=args.z)
        
    def forward_NAE(self, image):
        z, x_h = self.NAE(image)
        return z, x_h

    def test_NAE(self, z):
        x_h = self.NAE(z, test=True)
        return x_h

    def model_cuda_init(self):
        if torch.cuda.is_available():
            self.NAE = nn.DataParallel(self.NAE).cuda()

    def weight_cuda_init(self):
        self.NAE.apply(self.weights_init_normal)
        
    def weights_init_normal(self, m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            # if init_type == 'normal':
            #     torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            # elif init_type == 'xavier':
            #     torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
            # elif init_type == 'kaiming':
            #     torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def optimizer_init(self, args):
        self.optim_NAE = optim.Adam(self.NAE.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay) # lr, b1, b2, weight_decay
        # self.optim_NAE = optim.SGD(self.NAE.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    def learning_scheduler_init(self, args, mode):
        if mode == "NAE":
            self.lr_NAE = optim.lr_scheduler.MultiStepLR(self.optim_NAE, args.milestones, gamma=args.gamma, last_epoch=args.last_epoch)

    def load_state_dict(self, checkpoint, mode):
        if mode == "NAE":
            self.NAE.load_state_dict(checkpoint['weight'])
            self.optim_NAE.load_state_dict(checkpoint['optim'])


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
    if mode == "NAE":
        save_checkpoint({
            'epoch': epoch,
            'Best_Prec': best_prec_result,
            'model': state_info.NAE,
            'weight': state_info.NAE.state_dict(),
            'optim': state_info.optim_NAE.state_dict(),
        }, filename, directory)

    # print(state_info.optim_Disc.state_dict()['param_groups'])
    # optim = state_info.optim_Disc.state_dict()
    # print(optim['param_groups'])

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
