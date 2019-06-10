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
import pretrain
import dataset
import math


parser = argparse.ArgumentParser(description='PyTorch Cycle Domain Adaptation Training')
parser.add_argument('--sd', default='mnist', type=str, help='source dataset')
parser.add_argument('--td', default='usps', type=str, help='target dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epoch', default=10, type=int, metavar='N', help='number of total epoch to run')
parser.add_argument('--decay-epoch', default=30, type=int, metavar='N', help='epoch from which to start lr decay')
parser.add_argument('--Pepoch', default=10, type=int, metavar='N', help='Pretrain model number of total epoch to run')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent-size', type=int, default=20, help='dimension of latent z')
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

def loss_function(x_hat, x, mu, log_var, cls, y):
    BCE = criterion_BCE(x_hat, x.view(x.size(0), -1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    CE = args.CE * criterion(cls, y)
    return BCE + KLD + CE, BCE.item(), KLD.item(), CE.item()

def main():
    global args, best_prec_result
    
    start_epoch = 0
    utils.default_model_dir = args.dir
    start_time = time.time()

    Source_train_loader, Source_test_loader, src_ch, src_wh = dataset_selector(args.sd)
    Target_train_loader, Target_test_loader, tar_ch, tar_wh = dataset_selector(args.td)
    Src_sample, Tgt_sample = extract_sample(Source_train_loader, Target_train_loader)

    state_info = utils.model_optim_state_info()
    state_info.model_init(Img_S=[src_ch, src_wh], Img_T=[tar_ch, tar_wh]
                        , H=args.h, latent_size=args.latent_size, num_class=10)
    state_info.model_cuda_init()
    state_info.weight_init()
    state_info.optimizer_init(args)

    if cuda:
        print("USE", torch.cuda.device_count(), "GPUs!")
        cudnn.benchmark = True

    pretrain.pretrain(args, state_info, Source_train_loader, Source_test_loader, Src_sample)

    checkpoint = utils.load_checkpoint(utils.default_model_dir, is_last=True, is_source=False)    
    if not checkpoint:
        state_info.learning_scheduler_init(args)
    else:
        start_epoch = checkpoint['epoch'] + 1
        best_prec_result = checkpoint['Best_Prec']
        state_info.load_state_dict(checkpoint)
        state_info.learning_scheduler_init(args, load_epoch=start_epoch)

    for epoch in range(start_epoch, args.epoch):
        
        train(state_info, Target_train_loader, epoch)
        test(state_info, Target_test_loader, Src_sample, Tgt_sample, epoch)

        # if prec_result > best_prec_result:
        #     best_prec_result = prec_result
        #     filename = 'checkpoint_best.pth.tar'
        #     utils.save_target_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)

        filename = 'latest.pth.tar'
        utils.save_target_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)
        state_info.learning_step() 

    filename = 'target_final.pth.tar'
    utils.save_target_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)

    now = time.gmtime(time.time() - start_time)
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


def train(state_info, Target_train_loader, epoch): # all 

    utils.print_log('Type, Epoch, Batch, loss, BCE, KLD, CE')
    state_info.set_train_mode()
    correct = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)

    for it, (x, y) in enumerate(Target_train_loader):

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)
        x_hat_t, mu, log_var, z, clsT, clsS = state_info.forward(x)
        
        _, clsS = torch.max(clsS.data, 1)
        clsS = to_var(clsS, LongTensor)

        #  Train 
        state_info.optim_VAE_tgt.zero_grad()
        loss, BCE, KLD, CE = loss_function(x_hat_t, x, mu, log_var, clsT, clsS)
        loss.backward(retain_graph=True)
        state_info.optim_VAE_tgt.step()

        # mapping info of <y, cls_output> print
        
        #  Log Print
        total += float(clsT.size(0))
        _, predicted = torch.max(clsT.data, 1)

        correct += float(predicted.eq(clsS.data).cpu().sum())

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'
                  .format(epoch, it, loss.item(), BCE, KLD, CE, 100.*correct / total))
            print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'
                  .format(epoch, it, loss.item(), BCE, KLD, CE, 100.*correct / total))

    utils.print_log('')

def test(state_info, Target_test_loader, Src_sample, Tgt_sample, epoch):

    utils.print_log('Type, Epoch, Batch, Acc')
    state_info.set_test_mode()
    correct = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)

    for it, (x, y) in enumerate(Target_test_loader):

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)
        clsT, clsS, _, _ = state_info.forward(x, test=True)

        # mapping info of <y, cls_output> print
        clsS = torch.max(clsS.data, 1)[1]
        clsS = to_var(clsS, LongTensor)

        #  Log Print
        total += float(clsT.size(0))
        _, predicted = torch.max(clsT.data, 1)
        correct += float(predicted.eq(clsS.data).cpu().sum())
        
        if it % 10 == 0:
            utils.print_log('Test, {}, {}, {:.2f}'.format(epoch, it, 100.*correct / total))
            print('Test, {}, {}, {:.2f}'.format(epoch, it, 100.*correct / total))
    
    make_sample_image(state_info, Src_sample, Tgt_sample, epoch)
    utils.print_log('')

def make_sample_image(state_info, Src_sample, Tgt_sample, epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    # img_path1 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/SS'))
    img_path2 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/ST'))
    img_path3 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/TS'))
    img_path4 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/TT'))

    _, _, z = state_info.pretrain_forward(Src_sample, test=True)
    _, _T = state_info.forward_z(z)
    Src_sample, _T = to_data(Src_sample), to_data(_T)

    # SS = merge_images(Src_sample, S_)
    ST = merge_images(Src_sample, _T)

    _, _, T_, _S = state_info.forward(Tgt_sample, test=True)
    Tgt_sample, T_, _S = to_data(Tgt_sample), to_data(T_), to_data(_S)

    TT = merge_images(Tgt_sample, T_)
    TS = merge_images(Tgt_sample, _S)

    # save_image(SS.data, os.path.join(img_path1, '%d.png' % epoch), normalize=True)
    save_image(ST.data, os.path.join(img_path2, '%d.png' % epoch), normalize=True)
    save_image(TS.data, os.path.join(img_path3, '%d.png' % epoch), normalize=True)
    save_image(TT.data, os.path.join(img_path4, '%d.png' % epoch), normalize=True)

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
        return dataset.SVHN_loader(img_size=args.img_size)
    elif data == "usps":
        return dataset.usps_loader(img_size=args.img_size)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x, dtype):
    return Variable(x.type(dtype))

def extract_sample(Source_train_loader, Target_train_loader):

    data_zip = enumerate(zip(Source_train_loader, Target_train_loader))
    for step, ((Src_sample, _), (Tgt_sample, _)) in data_zip:
        Src_sample = to_var(Src_sample, FloatTensor)
        Tgt_sample = to_var(Tgt_sample, FloatTensor)
        break;

    return Src_sample, Tgt_sample

# def extract_sample(Source_train_loader, Target_train_loader):
#     Src_sample_iter = iter(Source_train_loader)
#     Tgt_sample_iter = iter(Target_train_loader)

#     Src_sample = Src_sample_iter.next()
#     Src_sample, Src_label = to_var(Src_sample[0], FloatTensor), Src_sample[1]

#     Tgt_sample = Tgt_sample_iter.next()
#     Tgt_sample, Tgt_label = to_var(Tgt_sample[0], FloatTensor), Tgt_sample[1]

#     return Src_sample, Src_label, Tgt_sample, Tgt_label

if __name__=='__main__':
    main()