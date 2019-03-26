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
import math

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--sd', default='mnist', type=str, help='source dataset')
parser.add_argument('--td', default='svhn', type=str, help='target dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent-dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--train-iters', type=int, default=40000, help='number of iteration')

parser.add_argument('--dir', default='./', type=str, help='default save directory')
parser.add_argument('--gpu', default='', type=str, help='Multi GPU ids to use.')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

rgb2grayWeights = [0.2989, 0.5870, 0.1140]
source_prediction_max_result = []
target_prediction_max_result = []
best_prec_result = 0

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def dataset_selector(dataset):
    if dataset == 'mnist':
        return utils.MNIST_loader()
    elif dataset == 'svhn':
        return utils.SVHN_loader()

def to_var(x):
    if cuda:
        x = x.cuda()
    return Variable(x)

def main():
    global args, best_prec_result
    args = parser.parse_args()
    
    if args.gpu != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    utils.default_model_dir = args.dir
    start_time = time.time()

    Source_train_loader, Source_test_loader = dataset_selector(args.sd)
    Target_train_loader, Target_test_loader = dataset_selector(args.td)

    state_info = utils.model_optim_state_info()
    state_info.model_init()
    state_info.model_cuda_init()

    if cuda:
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        print("USE", torch.cuda.device_count(), "GPUs!")
        cudnn.benchmark = True
    else:
        print("NO GPU")

    state_info.weight_cuda_init()
    state_info.optimizer_init(lr=args.lr, b1=args.b1, b2=args.b2, weight_decay=args.weight_decay)

    adversarial_loss = torch.nn.BCELoss()
    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = 0

    utils.default_model_dir
    filename = 'latest.pth.tar'
    checkpoint = utils.load_checkpoint(utils.default_model_dir)    
    if not checkpoint:
        pass
    else:
        start_epoch = checkpoint['epoch'] + 1
        best_prec_result = checkpoint['Best_Prec']
        state_info.load_state_dict(checkpoint)

    numEpochs = int(math.ceil(float(args.train_iters) / float(min(len(Source_train_loader), len(Target_train_loader)))))

    for epoch in range(numEpochs):
        # if epoch < 80:
        #     learning_rate = args.lr
        # elif epoch < 120:
        #     learning_rate = args.lr * 0.1
        # else:
        #     learning_rate = args.lr * 0.01
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = learning_rate

        train(state_info, Source_train_loader, Target_train_loader, criterion, adversarial_loss, epoch)
        prec_result = test(state_info, Source_test_loader, Target_test_loader, criterion, epoch)

        if prec_result > best_prec_result:
            best_prec_result = prec_result
            filename = 'checkpoint_best.pth.tar'
            utils.save_state_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)

        if epoch % 5 == 0:
            filename = 'latest.pth.tar'
            utils.save_state_checkpoint(state_info, best_prec_result, filename, utils.default_model_dir, epoch)     

    now = time.gmtime(time.time() - start_time)
    print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))


def train(state_info, Source_train_loader, Target_train_loader, criterion, adversarial_loss, epoch): # all 

    utils.print_log('Type, Epoch, Batch, GCsrc, GCtar, GRloss, DCsrc, DCtar, DRloss, ACCsrc, CSloss, ACCtar, CTloss')
    state_info.set_train_mode()
    correct_src = 0
    correct_target = 0
    total = 0

    for it, ((Source_data, y), (Target_data, _)) in enumerate(zip(Source_train_loader, Target_train_loader)):
        
        if Target_data.size(0) != Source_data.size(0):
            continue
        
        batch_size = Source_data.size(0)
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        y_one = torch.FloatTensor(batch_size, 10).zero_().scatter_(1, y.view(-1, 1), 1)

        Source_data, y, y_one = to_var(Source_data), to_var(y), to_var(y_one)
        Target_data = to_var(Target_data)

        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim))))

        # G - Creation

        state_info.optimizer_SG.zero_grad()
        state_info.optimizer_TG.zero_grad()

        img_gen_src = state_info.gen_src(z, y_one)
        loss_gen_src = adversarial_loss(state_info.disc_src(img_gen_src), valid)

        img_gen_target = state_info.gen_target(z, y_one)
        loss_gen_target = adversarial_loss(state_info.disc_target(img_gen_target), valid)

        loss_gen_src.backward()
        loss_gen_target.backward()
        
        # G - Representation

        black_img_gen_src, black_img_gen_target = img_gen_src, img_gen_target
        if black_img_gen_src.size(1) == 3:
            black_img_gen_src = rgb2grayWeights[0] * black_img_gen_src[:,0,:,:] + rgb2grayWeights[1] * black_img_gen_src[:,1,:,:] + rgb2grayWeights[2] * black_img_gen_src[:,2,:,:]
            black_img_gen_src.unsqueeze_(1)
        if black_img_gen_target.size(1) == 3:
            black_img_gen_target = rgb2grayWeights[0] * black_img_gen_target[:,0,:,:] + rgb2grayWeights[1] * black_img_gen_target[:,1,:,:] + rgb2grayWeights[2] * black_img_gen_target[:,2,:,:]
            black_img_gen_target.unsqueeze_(1)

        loss_rep_gen_src = adversarial_loss(state_info.disc_class(black_img_gen_src, y_one), valid)
        loss_rep_gen_target = adversarial_loss(state_info.disc_class(black_img_gen_target, y_one), valid)
        loss_rep_gen = (loss_rep_gen_src + loss_rep_gen_target) / 2

        loss_rep_gen.backward()

        state_info.optimizer_SG.step()
        state_info.optimizer_TG.step()

        # D - Creation

        state_info.optimizer_SD.zero_grad()
        state_info.optimizer_TD.zero_grad()

        loss_dis_src_real = adversarial_loss(state_info.disc_src(Source_data), valid)
        loss_dis_src_fake = adversarial_loss(state_info.disc_src(img_gen_src.detach()), fake)
        loss_dis_src = loss_dis_src_real + loss_dis_src_fake / 2

        loss_dis_target_real = adversarial_loss(state_info.disc_target(Target_data), valid)
        loss_dis_target_fake = adversarial_loss(state_info.disc_target(img_gen_target.detach()), fake)
        loss_dis_target = loss_dis_target_real + loss_dis_target_fake / 2

        loss_dis_src.backward()
        loss_dis_target.backward()

        state_info.optimizer_SD.step()
        state_info.optimizer_TD.step()

        # D - Representation

        state_info.optimizer_REP.zero_grad()

        black_Source_data = Source_data
        if black_Source_data.size(1) == 3:
            black_Source_data = rgb2grayWeights[0] * black_Source_data[:,0,:,:] + rgb2grayWeights[1] * black_Source_data[:,1,:,:] + rgb2grayWeights[2] * black_Source_data[:,2,:,:]
            black_Source_data.unsqueeze_(1)

        loss_rep_dis_src = adversarial_loss(state_info.disc_class(black_img_gen_src, y_one), fake)
        loss_rep_dis_target = adversarial_loss(state_info.disc_class(black_img_gen_target, y_one), fake)
        loss_rep_dis_real = adversarial_loss(state_info.disc_class(black_Source_data, y_one), valid)
        loss_rep_dis = (loss_rep_dis_src + loss_rep_dis_target + loss_rep_dis_real) / 3

        loss_rep_dis.backward()
        state_info.optimizer_REP.step()

        # Class Prediction
        # NO NEED : loss_criterion_src

        state_info.optimizer_CS.zero_grad()
        state_info.optimizer_CT.zero_grad()
        
        output_cls_gen_src = state_info.cls_src(img_gen_src)
        output_cls_gen_target = state_info.cls_target(img_gen_target)

        loss_criterion_src = criterion(output_cls_gen_src, y)
        loss_criterion_target = criterion(output_cls_gen_target, y)

        loss_criterion_src.backward()
        loss_criterion_target.backward()

        state_info.optimizer_CS.step()
        state_info.optimizer_CT.step()

        total += output_cls_gen_src.size(0)
        _, predicted_src = torch.max(output_cls_gen_src.data, 1)
        correct_src += predicted_src.eq(y.data).cpu().sum()

        _, predicted_target = torch.max(output_cls_gen_target.data, 1)
        correct_target += predicted_target.eq(y.data).cpu().sum()



        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}, {:.4f}, {:.2f}, {:.4f}'
                  .format(epoch, it, loss_gen_src.data[0], loss_gen_target.data[0], loss_rep_gen.data[0], loss_dis_src.data[0], loss_dis_target.data[0], loss_rep_dis.data[0]
                    , 100.*(correct_src / total), loss_criterion_src.data[0], 100.*(correct_target / total), loss_criterion_target.data[0]))
            print('Train, EP:{}, IT:{}, L)GS:{:.4f}, L)GT:{:.4f}, L)GR:{:.4f}, L)DS:{:.4f}, L)DT:{:.4f}, L)DR:{:.4f}, Acc)S:{:.2f}, L)S:{:.4f}, Acc)T:{:.2f}, L)T:{:.4f}'
                  .format(epoch, it, loss_gen_src.data[0], loss_gen_target.data[0], loss_rep_gen.data[0], loss_dis_src.data[0], loss_dis_target.data[0], loss_rep_dis.data[0]
                    , 100.*(correct_src / total), loss_criterion_src.data[0], 100.*(correct_target / total), loss_criterion_target.data[0]))

    utils.print_log('')

def test(state_info, Source_test_loader, Target_test_loader, criterion, epoch):
    
    utils.print_log('Type, Epoch, Batch, ACCsrc, CSloss, ACCtar, CTloss')
    state_info.set_test_mode()
    correct_src = 0
    correct_target = 0
    total = 0
    total_loss_src = 0
    total_loss_target = 0

    for it, ((Source_data, Source_y), (Target_data, Target_y)) in enumerate(zip(Source_test_loader, Target_test_loader)):

        if Target_data.size(0) != Source_data.size(0):
            continue
        
        batch_size = Source_data.size(0)

        Source_y_one = torch.FloatTensor(batch_size, 10).zero_().scatter_(1, Source_y.view(-1, 1), 1)
        Target_y_one = torch.FloatTensor(batch_size, 10).zero_().scatter_(1, Target_y.view(-1, 1), 1)

        Source_data, Source_y, Source_y_one = to_var(Source_data), to_var(Source_y), to_var(Source_y_one)
        Target_data, Target_y, Target_y_one = to_var(Target_data), to_var(Target_y), to_var(Target_y_one)

        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim))))
        
        img_gen_src = state_info.gen_src(z, Source_y_one)
        img_gen_target = state_info.gen_target(z, Target_y_one)

        output_cls_gen_src = state_info.cls_src(Source_data)
        output_cls_gen_target = state_info.cls_target(Target_data)

        loss_criterion_src = criterion(output_cls_gen_src, Source_y)
        loss_criterion_target = criterion(output_cls_gen_target, Target_y)

        total += output_cls_gen_src.size(0)
        _, predicted_src = torch.max(output_cls_gen_src.data, 1)
        correct_src += predicted_src.eq(Source_y.data).cpu().sum()

        _, predicted_target = torch.max(output_cls_gen_target.data, 1)
        correct_target += predicted_target.eq(Target_y.data).cpu().sum()

        total_loss_src += loss_criterion_src.data[0]
        total_loss_target += loss_criterion_target.data[0]
    
    make_sample_image(state_info, epoch, n_row=10) # img_gen_src, Source_y, img_gen_target, Target_y

    source_prediction_max_result.append(correct_src)
    target_prediction_max_result.append(correct_target)

    utils.print_log('Test, {}, {}, {:.2f}, {:.4f}, {:.2f}, {:.4f}'
          .format(epoch, it, 100.*(correct_src / total), total_loss_src / (it + 1), 100.*(correct_target / total), total_loss_target / (it + 1)))
    print('Test, EP:{}, IT:{}, Acc)S:{:.2f}, Loss)S:{:.4f}, Acc)T:{:.2f}, Loss)T:{:.4f}'
          .format(epoch, it, 100.*(correct_src / total), total_loss_src / (it + 1), 100.*(correct_target / total), total_loss_target / (it + 1)))

    return 100.*(correct_target / total)


def make_sample_image(state_info, epoch, n_row=10):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    img_path1 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/src'))
    img_path2 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/target'))

    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row**2, args.latent_dim))))

    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    img_gen_src = state_info.gen_src(z, labels)
    img_gen_target = state_info.gen_target(z, labels)
    save_image(img_gen_src.data, os.path.join(img_path1, '%d.png' % epoch), nrow=n_row, normalize=True)
    save_image(img_gen_target.data, os.path.join(img_path2, '%d.png' % epoch), nrow=n_row, normalize=True)


if __name__=='__main__':
    main()