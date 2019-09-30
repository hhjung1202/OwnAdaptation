import argparse
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import os
import torch.backends.cudnn as cudnn
import time
import utils
import dataset
import math

parser = argparse.ArgumentParser(description='PyTorch Noise Label Training')

parser.add_argument('--db', default='mnist', type=str, help='dataset selection')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--dir', default='./', type=str, help='default save directory')
parser.add_argument('--gpu', default='0', type=str, help='Multi GPU ids to use.')

parser.add_argument('--epoch', default=165, type=int, metavar='N', help='number of total epoch to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='learning gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

parser.add_argument('--img-size', type=int, default=32, help='input image width, height size')
parser.add_argument('--layer', type=int, default=20, help='[14, 20, 32, 44, 56, 110]')

parser.add_argument('--decay-epoch', default=100, type=int, metavar='N', help='epoch from which to start lr decay')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--cycle', type=float, default=10.0, help='Cycle Consistency Parameter')
parser.add_argument('--identity', type=float, default=5.0, help='Identity Consistency Parameter')
parser.add_argument('--cls', type=float, default=1.0, help='[A,y] -> G_AB -> G_BA -> [A_,y] Source Class Consistency Parameter')

best_prec_result = torch.tensor(0, dtype=torch.float32)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cuda = True if torch.cuda.is_available() else False

def main():
    global args, best_prec_result
    
    utils.default_model_dir = args.dir
    start_time = time.time()

    True_loader, Fake_loader, Noise_loader, Noise_Test_loader, All_loader, Test_loader, chIn, clsN = dataset_selector(args.db)
    args.chIn = chIn
    args.clsN = clsN
    args.milestones = [80,120]

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

    state_info.optimizer_init(args)


    # adversarial_loss = torch.nn.BCELoss()
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = 0

    checkpoint = utils.load_checkpoint(utils.default_model_dir)    
    if not checkpoint:
        args.last_epoch = -1
        state_info.learning_scheduler_init(args)
    else:
        start_epoch = checkpoint['epoch'] + 1
        best_prec_result = checkpoint['Best_Prec']
        state_info.load_state_dict(checkpoint)
        args.last_epoch = start_epoch
        state_info.learning_scheduler_init(args)

    for epoch in range(start_epoch, args.epoch):
        
        train(state_info, Source_train_loader, Target_train_loader, criterion_GAN, criterion_cycle, criterion_identity, criterion, epoch)
        prec_result = test(state_info, Source_test_loader, Target_test_loader, criterion, realA_sample, realB_sample, epoch)
        
        if prec_result > best_prec_result:
            best_prec_result = prec_result
            filename = 'checkpoint_best.pth.tar'
            utils.save_state_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)

        filename = 'latest.pth.tar'
        utils.save_state_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)
        state_info.learning_step() 

    now = time.gmtime(time.time() - start_time)
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


# def make_sample_image(state_info, epoch, realA_sample, realB_sample):
#     """Saves a grid of generated digits ranging from 0 to n_classes"""
#     # Sample noise
#     img_path1 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/src'))
#     img_path2 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/target'))

#     fake_B = state_info.G_AB(realA_sample)
#     fake_A = state_info.G_BA(realB_sample)

#     realA, fake_B = to_data(realA_sample), to_data(fake_B)
#     realB, fake_A = to_data(realB_sample), to_data(fake_A)

#     makeAtoB = merge_images(realA_sample, fake_B)
#     makeBtoA = merge_images(realB_sample, fake_A)

#     save_image(makeAtoB.data, os.path.join(img_path1, '%d.png' % epoch), normalize=True)
#     save_image(makeBtoA.data, os.path.join(img_path2, '%d.png' % epoch), normalize=True)

# def to_data(x):
#     """Converts variable to numpy."""
#     if torch.cuda.is_available():
#         x = x.cpu()
#     return x.data.numpy()

# def merge_images(sources, targets, row=10):
#     _, _, h, w = sources.shape
#     merged = np.zeros([3, row*h, row*w*2])
#     for idx, (s, t) in enumerate(zip(sources, targets)):
#         i = idx // row
#         j = idx % row
#         if i is row:
#             break
#         merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
#         merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t

#     return torch.from_numpy(merged)


def dataset_selector(data):
    if data == 'mnist':
        return dataset.MNIST_loader(img_size=args.img_size)
    elif data == 'svhn':
        return dataset.SVHN_loader(img_size=args.img_size)




# if (step+1) % self.sample_step == 0:
#     fake_svhn = self.g12(fixed_mnist)
#     fake_mnist = self.g21(fixed_svhn)
    
#     mnist, fake_mnist = self.to_data(fixed_mnist), self.to_data(fake_mnist)
#     svhn , fake_svhn = self.to_data(fixed_svhn), self.to_data(fake_svhn)
    
#     merged = self.merge_images(mnist, fake_svhn)
#     path = os.path.join(self.sample_path, 'sample-%d-m-s.png' %(step+1))
#     scipy.misc.imsave(path, merged)
#     print ('saved %s' %path)
    
#     merged = self.merge_images(svhn, fake_mnist)
#     path = os.path.join(self.sample_path, 'sample-%d-s-m.png' %(step+1))
#     scipy.misc.imsave(path, merged)
#     print ('saved %s' %path)



if __name__=='__main__':
    main()