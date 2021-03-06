import argparse
import torch
import os
import torch.backends.cudnn as cudnn
import time
import utils
from train import *
import dataset

parser = argparse.ArgumentParser(description='PyTorch Noise Label Training')

parser.add_argument('--db', default='cifar10', type=str, help='dataset selection')
parser.add_argument('--noise-rate', default=0.5, type=float, help='Noise rate')
parser.add_argument('--noise-type', default="sym", type=str, help='Noise type : sym, Asym')
parser.add_argument('-seed', default=1234, type=int, help='random seed')
parser.add_argument('--model', default='ResNet18_rot', type=str, help='ResNet18, ResNet34')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--dir', default='./', type=str, help='default save directory')
parser.add_argument('--gpu', default='0', type=str, help='Multi GPU ids to use.')
parser.add_argument('--epoch', default=300, type=int, metavar='N', help='number of total epoch to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='learning gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--img-size', type=int, default=32, help='input image width, height size')
parser.add_argument('--m', type=int, default=0, help='latent selection(0 to n)')
parser.add_argument('--case', default=0, type=int, metavar='N', help='case')
parser.add_argument('--n-labeled', default=5000, type=int, metavar='N', help='semi labeled')
parser.add_argument('--iteration', type=int, default=512, help='Iteration')
parser.add_argument('--style', default=0, type=int, metavar='N', help='post style')
parser.add_argument('-s', '--serial', nargs='+', type=int, help='Block component: 0, 1, 2, 3')
parser.add_argument('-w', '--weight', nargs='+', type=float, help='Weight Parameter 14')

parser.add_argument('--n', type=int, default=4, help='Style count')
parser.add_argument('--type', default='self', type=str, help='self, cls')
parser.add_argument('--loss', nargs='+', type=int, help='5 argumentation 0 or 1')



best_prec_result = torch.tensor(0, dtype=torch.float32)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cuda = True if torch.cuda.is_available() else False
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if args.noise_type == "sym":
    args.sym = True
else:
    args.sym = False

def main():
    global args, best_prec_result

    Train_loader, Test_loader, chIn, clsN = dataset_selector()

    args.chIn = chIn
    args.clsN = clsN
    args.milestones = [80,120]
    
    state_info = utils.model_optim_state_info()
    state_info.model_init(args)
    state_info.model_cuda_init()

    if cuda:
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        print("USE", torch.cuda.device_count(), "GPUs!")
        state_info.weight_cuda_init()
        cudnn.benchmark = True
    else:
        print("NO GPU")

    state_info.optimizer_init(args)
    train_MEM(args, state_info, Train_loader, Test_loader)

def dataset_selector():
    # return dataset.cifar100_loader(args)
    return dataset.cifar10_loader(args)

if __name__=='__main__':
    main()