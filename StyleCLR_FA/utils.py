import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        if args.model == "ResNet18":
            self.model = ResNet18(serial=args.serial, style_out=args.style, num_classes=args.clsN, n=args.n, L_type=args.type)
        elif args.model == "ResNet34":
            self.model = ResNet34(serial=args.serial, style_out=args.style, num_classes=args.clsN, n=args.n, L_type=args.type)
        elif args.model == "vgg":
            enc = vgg
            dec = decoder
            self.model = Net(enc, dec)


    # def forward(self, x, y, u_x):
    #     loss_s, JS_loss, loss_u, style_loss, content_loss = self.model(x, y, u_x)
    #     return loss_s, JS_loss, loss_u, style_loss, content_loss

    def forward(self, x):
        loss_a, loss_c, loss_s = self.model(x)
        return loss_a, loss_c, loss_s

    def test(self, x):
        content, style, recon, adain = self.model(x, test=True)
        return content, style, recon, adain

    def model_cuda_init(self):
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model).cuda()

    def weight_cuda_init(self):
        self.model.apply(self.weights_init_normal)
        
    def weights_init_normal(self, m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            # torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
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
        # self.optim_model = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay) # lr, b1, b2, weight_decay
        self.optim_model = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    def learning_scheduler_init(self, args, mode=None):
        # if mode == "NAE":
        self.lr_model = optim.lr_scheduler.MultiStepLR(self.optim_model, args.milestones, gamma=args.gamma, last_epoch=args.last_epoch)

    def load_state_dict(self, checkpoint, mode=None):
        # if mode == "NAE":
        self.model.load_state_dict(checkpoint['weight'])
        self.optim_model.load_state_dict(checkpoint['optim'])


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def save_state_checkpoint(state_info, best_prec_result, epoch, filename, directory, mode=None):
    # if mode == "NAE":
    save_checkpoint({
        'epoch': epoch,
        'Best_Prec': best_prec_result,
        'model': state_info.model,
        'weight': state_info.model.state_dict(),
        'optim': state_info.optim_model.state_dict(),
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
