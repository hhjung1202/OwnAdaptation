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
parser.add_argument('-seed', default=1234, type=int, help='random seed')
parser.add_argument('--model', default='MobileNetV2', type=str, help='ResNet18, ResNet34')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--dir', default='./', type=str, help='default save directory')
parser.add_argument('--gpu', default='0', type=str, help='Multi GPU ids to use.')
parser.add_argument('--epoch', default=300, type=int, metavar='N', help='number of total epoch to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='learning gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--img-size', type=int, default=32, help='input image width, height size')

parser.add_argument('--gate', default='True', type=str, help='True, False')
parser.add_argument('--iter', default=1, type=int, help='iter per epoch')


best_prec_result = torch.tensor(0, dtype=torch.float32)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cuda = True if torch.cuda.is_available() else False
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if args.gate == "True":
    args.use_gate = True
else:
    args.use_gate = False

def main():
    global args, best_prec_result

    Train_loader, Test_loader, chIn, clsN = dataset_selector(args.db)

    args.chIn = chIn
    args.clsN = clsN
    args.milestones = [150,225]
    
    state_info = utils.model_optim_state_info()
    state_info.model_init(args)
    utils.init_learning(state_info.model.module)

    if cuda:
        print("USE", torch.cuda.device_count(), "GPUs!")
        cudnn.benchmark = True
    else:
        print("NO GPU")

    state_info.optimizer_init(args)
    train_Epoch(args, state_info, Train_loader, Test_loader)

def dataset_selector(db):
    if db == "cifar10":
        return dataset.cifar10_loader(args)
    else:
        return dataset.cifar100_loader(args)

if __name__=='__main__':
    main()