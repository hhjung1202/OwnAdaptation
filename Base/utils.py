import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import *
import torch.optim as optim
import csv
import os

default_model_dir = "./"
c = None
str_w = ''
csv_file_name = 'weight.csv'


class model_optim_state_info(object):
    def __init__(self):
        pass

    def model_init(self):
        self.gen_src = Generator(out_dim=1) # input : [z, y]
        self.disc_src = Discriminator(x_dim=1) # input : x_src, gen_src
        
        self.gen_target = Generator(out_dim=3) # input : [z, y]
        self.disc_target = Discriminator(x_dim=3) # input : x_target, gen_target

        self.disc_class = Discriminator_rep(x_dim=1) # [x_src, y], [gen_src, y], [gen_target(channel make 1), y]

        self.cls_src = Classifier(x_dim=1) # input: gen_src
        self.cls_target = Classifier(x_dim=3) # input: gen_target
        
    def model_cuda_init(self):
        if torch.cuda.is_available():
            self.gen_src = nn.DataParallel(self.gen_src).cuda()
            self.disc_src = nn.DataParallel(self.disc_src).cuda()
            self.gen_target = nn.DataParallel(self.gen_target).cuda()
            self.disc_target = nn.DataParallel(self.disc_target).cuda()
            self.disc_class = nn.DataParallel(self.disc_class).cuda()
            self.cls_src = nn.DataParallel(self.cls_src).cuda()
            self.cls_target = nn.DataParallel(self.cls_target).cuda()

    def weight_cuda_init(self):
        self.gen_src.apply(self.weights_init_normal)
        self.disc_src.apply(self.weights_init_normal)
        self.gen_target.apply(self.weights_init_normal)
        self.disc_target.apply(self.weights_init_normal)
        self.disc_class.apply(self.weights_init_normal)
        self.cls_src.apply(self.weights_init_normal)
        self.cls_target.apply(self.weights_init_normal)

    def weights_init_normal(self, m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def optimizer_init(self, lr, b1, b2, weight_decay):
        self.optimizer_SG = optim.Adam(self.gen_src.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_SD = optim.Adam(self.disc_src.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_TG = optim.Adam(self.gen_target.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_TD = optim.Adam(self.disc_target.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_REP = optim.Adam(self.disc_class.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_CS = optim.Adam(self.cls_src.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_CT = optim.Adam(self.cls_src.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)

    def set_train_mode(self):
        self.gen_src.train()
        self.disc_src.train()
        self.gen_target.train()
        self.disc_target.train()
        self.disc_class.train()
        self.cls_src.train()
        self.cls_target.train()

    def set_test_mode(self):
        self.gen_src.eval()
        self.disc_src.eval()
        self.gen_target.eval()
        self.disc_target.eval()
        self.disc_class.eval()
        self.cls_src.eval()
        self.cls_target.eval()

    def load_state_dict(self, checkpoint):
        self.gen_src.load_state_dict(checkpoint['SGdict'])
        self.disc_src.load_state_dict(checkpoint['SDdict'])
        self.gen_target.load_state_dict(checkpoint['TGdict'])
        self.disc_target.load_state_dict(checkpoint['TDdict'])
        self.disc_class.load_state_dict(checkpoint['REPdict'])
        self.cls_src.load_state_dict(checkpoint['CSdict'])
        self.cls_target.load_state_dict(checkpoint['CTdict'])

        self.optimizer_SG.load_state_dict(checkpoint['SGoptimizer'])
        self.optimizer_SD.load_state_dict(checkpoint['SDoptimizer'])
        self.optimizer_TG.load_state_dict(checkpoint['TGoptimizer'])
        self.optimizer_TD.load_state_dict(checkpoint['TDoptimizer'])
        self.optimizer_REP.load_state_dict(checkpoint['REPoptimizer'])
        self.optimizer_CS.load_state_dict(checkpoint['CSoptimizer'])
        self.optimizer_CT.load_state_dict(checkpoint['CToptimizer'])





def get_num_gen(gen):
    return sum(1 for x in gen)

def is_leaf(model):
    return get_num_gen(model.children()) == 0 

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# def weights_init_normal(m):
#     if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif isinstance(m, nn.BatchNorm2d):
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)

def save_state_checkpoint(state_info, best_prec_result, filename, directory, epoch):
    save_checkpoint({
        'epoch': epoch,
        'Best_Prec': best_prec_result,

        'SGmodel': state_info.gen_src,
        'SGdict': state_info.gen_src.state_dict(),
        'SGoptimizer': state_info.optimizer_SG.state_dict(),

        'SDmodel': state_info.disc_src,
        'SDdict': state_info.disc_src.state_dict(),
        'SDoptimizer': state_info.optimizer_SD.state_dict(),

        'TGmodel': state_info.gen_target,
        'TGdict': state_info.gen_target.state_dict(),
        'TGoptimizer': state_info.optimizer_TG.state_dict(),

        'TDmodel': state_info.disc_target,
        'TDdict': state_info.disc_target.state_dict(),
        'TDoptimizer': state_info.optimizer_TD.state_dict(),

        'REPmodel': state_info.disc_class,
        'REPdict': state_info.disc_class.state_dict(),
        'REPoptimizer': state_info.optimizer_REP.state_dict(),

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

def SVHN_loader(img_size=64, batchSize=128):
    print("SVHN Data Loading ...")

    rgb2grayWeights = [0.2989, 0.5870, 0.1140]
    train_dataset = datasets.SVHN(root='/home/hhjung/hhjung/SVHN/', split='train', 
                                        transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()
                                            , transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                                        download=True)

    test_dataset = datasets.SVHN(root='/home/hhjung/hhjung/SVHN/', split='test', 
                                        transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()
                                            , transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                                        download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)
    return train_loader, test_loader

    
def MNIST_loader(img_size=64, batchSize=128):
    print("MNIST Data Loading ...")

    train_dataset = datasets.MNIST(root='/home/hhjung/hhjung/MNIST/', train=True,
                                        transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()
                                            , transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                                        download=True)

    test_dataset = datasets.MNIST(root='/home/hhjung/hhjung/MNIST/', train=False,
                                       transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()
                                            , transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                                       download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)
    return train_loader, test_loader


def cifar10_loader():
    batch_size = 128
    rgb2grayWeights = [0.2989, 0.5870, 0.1140]

    print("cifar10 Data Loading ...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root='../hhjung/cifar10/', train=True, transform=transform_train, download=True)

    test_dataset = datasets.CIFAR10(root='../hhjung/cifar10/', train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def cifar100_loader():
    batch_size = 128
    print("cifar100 Data Loading ...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std=(0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std=(0.2675, 0.2565, 0.2761))
    ])

    train_dataset = datasets.CIFAR100(root='../hhjung/cifar100/',
                                     train=True,
                                     transform=transform_train,
                                     download=True)

    test_dataset = datasets.CIFAR100(root='../hhjung/cifar100/',
                                    train=False,
                                    transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    return train_loader, test_loader