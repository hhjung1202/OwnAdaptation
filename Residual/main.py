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
parser.add_argument('--sd', default='svhn', type=str, help='source dataset')
parser.add_argument('--td', default='mnist', type=str, help='target dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epoch', default=400, type=int, metavar='N', help='number of total epoch to run')
parser.add_argument('--decay-epoch', default=100, type=int, metavar='N', help='epoch from which to start lr decay')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent-dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img-size', type=int, default=32, help='input image width, height size')
parser.add_argument('--max-buffer', type=int, default=8196, help='Fake GAN Buffer Image')

parser.add_argument('--dir', default='./', type=str, help='default save directory')
parser.add_argument('--gpu', default='0', type=str, help='Multi GPU ids to use.')

parser.add_argument('--cycle', type=float, default=1.0, help='Cycle Consistency Parameter')
parser.add_argument('--identity', type=float, default=1.0, help='Identity Consistency Parameter')
parser.add_argument('--cls', type=float, default=1.0, help='[A,y] -> G_AB -> G_BA -> [A_,y] Source Class Consistency Parameter')
parser.add_argument('--gen', type=float, default=1.0, help='Target Generator loss weight')
parser.add_argument('--gen2', type=float, default=1.0, help='Source Generator loss weight')
parser.add_argument('--dis', type=float, default=1.0, help='Target Discriminator loss weight')
parser.add_argument('--dis2', type=float, default=1.0, help='Source Discriminator loss weight')
parser.add_argument('--recon', type=float, default=0.0, help='Discriminator loss weight')
parser.add_argument('--recon2', type=float, default=0.0, help='Discriminator loss weight')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

source_prediction_max_result = []
target_prediction_max_result = []
best_prec_result = torch.tensor(0, dtype=torch.float32)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

fake_S_buffer = utils.ImagePool(max_size=args.max_buffer)
fake_T_buffer = utils.ImagePool(max_size=args.max_buffer)

# adversarial_loss = torch.nn.BCELoss()
criterion_GAN = torch.nn.MSELoss()
criterion_L1 = torch.nn.L1Loss()
criterion_Recov = torch.nn.MSELoss()
criterion = nn.CrossEntropyLoss().cuda()

def main():
    global args, best_prec_result
    
    utils.default_model_dir = args.dir
    start_time = time.time()

    Source_train_loader, Source_test_loader = dataset_selector(args.sd)
    Target_train_loader, Target_test_loader = dataset_selector(args.td)
    Target_shuffle_loader, _ = dataset_selector(args.td)

    state_info = utils.model_optim_state_info()
    state_info.model_init()
    state_info.model_cuda_init()

    if cuda:
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        print("USE", torch.cuda.device_count(), "GPUs!")
        state_info.weight_cuda_init()
        cudnn.benchmark = True
    else:
        print("NO GPU")

    state_info.optimizer_init(lr=args.lr, b1=args.b1, b2=args.b2, weight_decay=args.weight_decay)

    start_epoch = 0

    checkpoint = utils.load_checkpoint(utils.default_model_dir)    
    if not checkpoint:
        state_info.learning_scheduler_init(args)
    else:
        start_epoch = checkpoint['epoch'] + 1
        best_prec_result = checkpoint['Best_Prec']
        state_info.load_state_dict(checkpoint)
        state_info.learning_scheduler_init(args, load_epoch=start_epoch)

    realS_sample_iter = iter(Source_train_loader)
    realT_sample_iter = iter(Target_train_loader)

    realS_sample = realS_sample_iter.next()
    realT_sample = to_var(realT_sample_iter.next()[0], FloatTensor)
    realS_sample, realS_y = to_var(realS_sample[0], FloatTensor), realS_sample[1]

    for epoch in range(args.epoch):
        
        train(state_info, Source_train_loader, Target_train_loader, Target_shuffle_loader, epoch)
        test(state_info, realS_sample, realS_y, realT_sample, epoch)
        # if prec_result > best_prec_result:
        #     best_prec_result = prec_result
        #     filename = 'checkpoint_best.pth.tar'
        #     utils.save_state_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)

        filename = 'latest.pth.tar'
        utils.save_state_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)
        state_info.learning_step() 

    now = time.gmtime(time.time() - start_time)
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


def train(state_info, Source_train_loader, Target_train_loader, Target_shuffle_loader, epoch): # all 

    utils.print_log('Type, Epoch, Batch, G-GAN-T, G-GAN-S, G-RECON, D-Target, D-Source')
    
    state_info.set_train_mode()

    for it, ((real_S, y), (real_T, _), (shuffle_T, _)) in enumerate(zip(Source_train_loader, Target_train_loader, Target_shuffle_loader)):
        
        if real_T.size(0) != real_S.size(0):
            continue
        
        batch_size = real_S.size(0)
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        y_one = torch.FloatTensor(batch_size, 10).zero_().scatter_(1, y.view(-1, 1), 1)

        real_S, y, y_one = to_var(real_S, FloatTensor), to_var(y, LongTensor), to_var(y_one, FloatTensor)
        real_T, shuffle_T = to_var(real_T, FloatTensor), to_var(shuffle_T, FloatTensor)

        # -----------------------
        #  Train Generator
        # -----------------------

        state_info.optim_G_Residual.zero_grad()
        fake_T, fake_S = state_info.forward(shuffle_T, y_one)

        loss_GAN_T = args.gen * criterion_GAN(state_info.D_tgt(fake_T), valid)
        loss_GAN_S = args.gen2 * criterion_GAN(state_info.D_src(fake_S, y_one), valid)
        loss_Recon = criterion_Recov(fake_T, shuffle_T)
        loss_Recon2 = criterion_L1(fake_S, real_S)
        loss_G = loss_GAN_T + loss_GAN_S + args.recon * loss_Recon + args.recon2 * loss_Recon2

        loss_G.backward(retain_graph=True)
        state_info.optim_G_Residual.step()

        # -----------------------
        #  Train Discriminator
        # -----------------------

        state_info.optim_D_tgt.zero_grad()
        state_info.optim_D_src.zero_grad()

        # fake_T_ = fake_T_buffer.query(fake_T)
        loss_real_T = criterion_GAN(state_info.D_tgt(real_T), valid)
        loss_fake_T = criterion_GAN(state_info.D_tgt(fake_T.detach()), fake)
        loss_real_S = criterion_GAN(state_info.D_src(real_S, y_one), valid)
        loss_fake_S = criterion_GAN(state_info.D_src(fake_S.detach(), y_one), fake)

        loss_Target = args.dis * (loss_real_T + loss_fake_T)
        loss_Source = args.dis2 * (loss_real_S + loss_fake_S)

        loss_Target.backward()
        loss_Source.backward()

        state_info.optim_D_tgt.step()
        state_info.optim_D_src.step()

        # -----------------------
        #  Log Print
        # -----------------------

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'
                  .format(epoch, it, loss_GAN_T.item(), loss_GAN_S.item(), loss_Recon.item(), loss_Recon2.item(), loss_Target.item(), loss_Source.item()))

            print('Train, {}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'
                  .format(epoch, it, loss_GAN_T.item(), loss_GAN_S.item(), loss_Recon.item(), loss_Recon2.item(), loss_Target.item(), loss_Source.item()))

    utils.print_log('')

def test(state_info, realS_sample, realS_y, realT_sample, epoch):

    realS_y = torch.FloatTensor(realS_sample.size(0), 10).zero_().scatter_(1, realS_y.view(-1, 1), 1)
    realS_y = to_var(realS_y, FloatTensor)
    
    state_info.set_test_mode()
    make_sample_image(state_info, epoch, realS_sample, realS_y, realT_sample) # img_gen_src, Source_y, img_gen_target, Target_y

def make_sample_image(state_info, epoch, realS_sample, realS_y, realT_sample):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    img_path1 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/resS'))
    img_path2 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/resT'))
    img_path3 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/cross'))

    fake_T, fake_S = state_info.forward(realT_sample, realS_y)
    fake_T, fake_S = to_data(fake_T), to_data(fake_S)

    residual1 = merge_images(realS_sample, fake_S)
    residual2 = merge_images(realT_sample, fake_T)
    residual3 = merge_images(realS_sample, fake_T)

    save_image(residual1.data, os.path.join(img_path1, '%d.png' % epoch), normalize=True)
    save_image(residual2.data, os.path.join(img_path2, '%d.png' % epoch), normalize=True)
    save_image(residual3.data, os.path.join(img_path3, '%d.png' % epoch), normalize=True)

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

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x, dtype):
    return Variable(x.type(dtype))

if __name__=='__main__':
    main()