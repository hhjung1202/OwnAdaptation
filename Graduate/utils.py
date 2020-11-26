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

def get_num_gen(gen):
    return sum(1 for x in gen)

def is_leaf(model):
    return get_num_gen(model.children()) == 0

def init_learning(model):
    for child in model.children():
        if hasattr(child, 'phase'):
            turn_off_learning(child)
        elif is_leaf(child):
            if hasattr(child, 'weight'):
                child.weight.requires_grad = True
                # print('True', child)
        else:
            init_learning(child)

def turn_off_learning(model):
    if is_leaf(model):
        if hasattr(model, 'weight'):
            model.weight.requires_grad = False
            # print('False', model)
        return

    for child in model.children():
        if is_leaf(child):
            if hasattr(child, 'weight'):
                child.weight.requires_grad = False
                # print('False', child)
        else:
            turn_off_learning(child)


def switching_learning(model):
    if is_leaf(model):
        if hasattr(model, 'weight'):
            if model.weight.requires_grad:
                model.weight.requires_grad = False
                # print('False', model)
            else:
                model.weight.requires_grad = True
                # print('True', model)
        return
    
    for child in model.children():

        if is_leaf(child):
            if hasattr(child, 'weight'):
                if child.weight.requires_grad:
                    child.weight.requires_grad = False
                    # print('False', child)
                else:
                    child.weight.requires_grad = True
                    # print('True', child)
        else:
            switching_learning(child)

class model_optim_state_info(object):
    def __init__(self):
        pass

    def model_init(self, args):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        
        self.init_lr = args.lr
        if args.model == "MobileNetV2":
            self.model = MobileNetV2(use_gate=args.use_gate, num_classes=args.clsN)
        elif args.model == "DenseNet":
            self.model = DenseNet(growth_rate=args.growth, num_init_features=args.init, num_classes=args.clsN, is_bottleneck=True, layer=args.layer)
        elif args.model == "DenseNet_Base":
            self.model = DenseNet_Base(growth_rate=args.growth, num_init_features=args.init, num_classes=args.clsN, is_bottleneck=True, layer=args.layer)
        elif args.model == "DenseNet_SE":
            self.model = DenseNet_SE(growth_rate=args.growth, num_init_features=args.init, num_classes=args.clsN, is_bottleneck=True, layer=args.layer)

        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model).cuda()

    def forward(self, x):
        output = self.model(x)
        return output

    def weight_init(self):
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
        self.optim_model = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    def learning_scheduler_init(self, args, mode=None):
        # if mode == "NAE":
        # self.lr_model = optim.lr_scheduler.CosineAnnealingLR(self.optim_model, T_max=args.epoch, eta_min=2e-6, last_epoch=-1)
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

def load_checkpoint(directory, is_last=False):

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
