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


parser = argparse.ArgumentParser(description='PyTorch Cycle Domain Adaptation Training')
parser.add_argument('--sd', default='mnist', type=str, help='source dataset')
parser.add_argument('--td', default='usps', type=str, help='target dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epoch', default=90, type=int, metavar='N', help='number of total epoch to run')
parser.add_argument('--decay-epoch', default=30, type=int, metavar='N', help='epoch from which to start lr decay')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent-size', type=int, default=100, help='dimension of latent z')
parser.add_argument('--h', type=int, default=400, help='dimension of hidden layer')
parser.add_argument('--img-size', type=int, default=28, help='input image width, height size')

parser.add_argument('--MSE', type=float, default=1.0, help='MSE loss parameter')
parser.add_argument('--KLD', type=float, default=1.0, help='KLD loss parameter')
parser.add_argument('--CE', type=float, default=1.0, help='CE loss parameter')

parser.add_argument('--dir', default='./', type=str, help='default save directory')
parser.add_argument('--gpu', default='0', type=str, help='Multi GPU ids to use.')

source_prediction_max_result = []
target_prediction_max_result = []
best_prec_result = torch.tensor(0, dtype=torch.float32)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(args.seed)

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

criterion_BCE = torch.nn.BCELoss(reduction='sum')
criterion = torch.nn.CrossEntropyLoss()

def loss_function(x_hat, x, mu, log_var):
    BCE = criterion_BCE(x_hat, x.view(x.size(0), -1))
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD, BCE.item(), KLD.item()

def main():
    global args, best_prec_result
    
    start_epoch = 0
    utils.default_model_dir = args.dir
    start_time = time.time()

    train_loader, test_loader, ch, wh = dataset_selector(args.sd)
    sample = extract_sample(train_loader)

    state_info = utils.model_optim_state_info()
    state_info.model_init(Img=[ch, wh], H=args.h, latent_size=args.latent_size, num_class=10)
    state_info.model_cuda_init()
    state_info.weight_init()
    state_info.optimizer_init(args)

    if cuda:
        print("USE", torch.cuda.device_count(), "GPUs!")
        cudnn.benchmark = True

    state_info.learning_scheduler_init(args)

    for epoch in range(start_epoch, args.epoch):
        
        train(state_info, train_loader, epoch)
        test(state_info, test_loader, sample, epoch)

        state_info.learning_step() 

    now = time.gmtime(time.time() - start_time)
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))

def train(state_info, train_loader, epoch): # all 

    utils.print_log('Type, Epoch, Batch, loss, BCE, KLD, CE')
    state_info.set_train_mode()
    correct = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)

    for it, (x, y) in enumerate(train_loader):

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)
        x_hat, mu, log_var, z = state_info.forward(x)
        
        #  Train 
        state_info.optim_VAE.zero_grad()
        loss, BCE, KLD = loss_function(x_hat, x, mu, log_var)
        loss.backward(retain_graph=True)
        state_info.optim_VAE.step()

        # mapping info of <y, cls_output> print
        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}'
                  .format(epoch, it, loss.item(), BCE, KLD))
            print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}'
                  .format(epoch, it, loss.item(), BCE, KLD))

    utils.print_log('')

def test(state_info, test_loader, sample, epoch):

    make_sample_image(state_info, sample, epoch)
    utils.print_log('')

def make_sample_image(state_info, sample, epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    img_path = utils.make_directory(os.path.join(utils.default_model_dir, 'image'))
    sample_hat, _, _, _ = state_info.forward(sample)
    sample, sample_hat = to_data(sample), to_data(sample_hat)
    image = merge_images(sample, sample_hat)
    save_image(image.data, os.path.join(img_path, '%d.png' % epoch), normalize=True)

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

def dataset_selector(data):
    if data == 'mnist':
        return dataset.MNIST_loader(img_size=args.img_size)
    elif data == 'svhn':
        return dataset.SVHN_loader(img_size=32)
    elif data == "usps":
        return dataset.usps_loader(img_size=args.img_size)
    elif data == "mnistm":
        return dataset.MNIST_M_loader(img_size=args.img_size)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x, dtype):
    return Variable(x.type(dtype))

def extract_sample(train_loader):

    for step, (sample, _) in enumerate(train_loader):
        sample = to_var(sample, FloatTensor)
        break;
    return sample

if __name__=='__main__':
    main()