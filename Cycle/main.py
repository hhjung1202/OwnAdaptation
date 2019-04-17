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
parser.add_argument('--td', default='svhn', type=str, help='target dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epoch', default=100, type=int, metavar='N', help='number of total epoch to run')
parser.add_argument('--decay-epoch', default=50, type=int, metavar='N', help='epoch from which to start lr decay')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent-dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img-size', type=int, default=32, help='input image width, height size')
parser.add_argument('--max-buffer', type=int, default=1024, help='Fake GAN Buffer Image')

parser.add_argument('--dir', default='./', type=str, help='default save directory')
parser.add_argument('--gpu', default='0', type=str, help='Multi GPU ids to use.')

parser.add_argument('--cycle', type=float, default=1.0, help='Cycle Consistency Parameter')
parser.add_argument('--identity', type=float, default=1.0, help='Identity Consistency Parameter')
parser.add_argument('--cls', type=float, default=1.0, help='[A,y] -> G_AB -> G_BA -> [A_,y] Source Class Consistency Parameter')

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

fake_A_buffer = utils.ImagePool(max_size=args.max_buffer)
fake_B_buffer = utils.ImagePool(max_size=args.max_buffer)

def main():
    global args, best_prec_result
    
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
        state_info.weight_cuda_init()
        cudnn.benchmark = True
    else:
        print("NO GPU")

    state_info.optimizer_init(lr=args.lr, b1=args.b1, b2=args.b2, weight_decay=args.weight_decay)


    # adversarial_loss = torch.nn.BCELoss()
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = 0

    checkpoint = utils.load_checkpoint(utils.default_model_dir)    
    if not checkpoint:
        state_info.learning_scheduler_init(args)
    else:
        start_epoch = checkpoint['epoch'] + 1
        best_prec_result = checkpoint['Best_Prec']
        state_info.load_state_dict(checkpoint)
        state_info.learning_scheduler_init(args, load_epoch=start_epoch)

    realA_sample_iter = iter(Source_train_loader)
    realB_sample_iter = iter(Target_train_loader)

    realA_sample = to_var(realA_sample_iter.next()[0], FloatTensor)
    realB_sample = to_var(realB_sample_iter.next()[0], FloatTensor)

    for epoch in range(args.epoch):
        
        # train(state_info, Source_train_loader, Target_train_loader, criterion_GAN, criterion_cycle, criterion_identity, criterion, epoch)
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


def train(state_info, Source_train_loader, Target_train_loader, criterion_GAN, criterion_cycle, criterion_identity, criterion, epoch): # all 

    utils.print_log('Type, Epoch, Batch, G-GAN, G-CYCLE, G-ID, G-CLASS, D-A, D-B, accREAL, ~loss, accRECOV, ~loss, accTAR, ~loss')

    state_info.set_train_mode()
    correct_real = torch.tensor(0, dtype=torch.float32)
    correct_recov = torch.tensor(0, dtype=torch.float32)
    correct_target = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)

    for it, ((real_A, y), (real_B, _)) in enumerate(zip(Source_train_loader, Target_train_loader)):
        
        if real_B.size(0) != real_A.size(0):
            continue
        
        if real_A.size(1) == 1:
            real_A = torch.cat([real_A, real_A, real_A], 1)

        if real_B.size(1) == 1:
            real_B = torch.cat([real_B, real_B, real_B], 1)
        
        batch_size = real_A.size(0)
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.latent_dim))))

        real_A, y = to_var(real_A, FloatTensor), to_var(y, LongTensor)
        real_B = to_var(real_B, FloatTensor)

        # -----------------------
        #  Train Source Classifier
        # -----------------------

        state_info.optimizer_CS.zero_grad()
        output_cls_real = state_info.cls_src(real_A) # Classifier
        loss_cls_clear = criterion(output_cls_real, y)
        loss_cls_clear.backward()
        state_info.optimizer_CS.step()

        # -----------------------
        #  Train Generator AB and BA
        # -----------------------

        state_info.optimizer_G_AB.zero_grad()
        state_info.optimizer_G_BA.zero_grad()

        # Identity loss
        loss_idt_A = criterion_identity(state_info.G_BA(real_A), real_A)
        loss_idt_B = criterion_identity(state_info.G_AB(real_B, z)[0], real_B)

        loss_identity = args.identity * (loss_idt_A + loss_idt_B) / 2

        # GAN loss
        fake_B, _, _ = state_info.G_AB(real_A, z)
        loss_GAN_AB = criterion_GAN(state_info.D_B(fake_B), valid)
        fake_A = state_info.G_BA(real_B)
        loss_GAN_BA = criterion_GAN(state_info.D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = state_info.G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B, _, _ = state_info.G_AB(fake_A, z)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = args.cycle * (loss_cycle_A + loss_cycle_B) / 2

        # Class Consistency
        output_cls_recov = state_info.cls_src(recov_A) # Classifier
        loss_cls_recov = args.cls * criterion(output_cls_recov, y)

        # Total loss
        loss_G = loss_GAN + loss_cycle + loss_identity + loss_cls_recov

        loss_G.backward(retain_graph=True)
        state_info.optimizer_G_AB.step()
        state_info.optimizer_G_BA.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        state_info.optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(state_info.D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.query(fake_A)
        loss_fake = criterion_GAN(state_info.D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        state_info.optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        state_info.optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(state_info.D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.query(fake_B)
        loss_fake = criterion_GAN(state_info.D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        state_info.optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # -----------------------
        #  Train Target Classifier
        # -----------------------

        state_info.optimizer_CT.zero_grad()
        output_cls_target = state_info.cls_target(fake_B) # Classifier
        loss_cls_fake = criterion(output_cls_target, y)
        loss_cls_fake.backward()
        state_info.optimizer_CT.step()

        # -----------------------
        #  Log Print
        # -----------------------

        # output_cls_real, output_cls_recov, output_cls_target, 

        total += float(batch_size)
        _, predicted_real = torch.max(output_cls_real.data, 1)
        correct_real += float(predicted_real.eq(y.data).cpu().sum())

        _, predicted_recov = torch.max(output_cls_recov.data, 1)
        correct_recov += float(predicted_recov.eq(y.data).cpu().sum())

        _, predicted_target = torch.max(output_cls_target.data, 1)
        correct_target += float(predicted_target.eq(y.data).cpu().sum())

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}, {:.4f}, {:.2f}, {:.4f}, {:.2f}, {:.4f}'
                  .format(epoch, it, loss_GAN.item(), loss_cycle.item(), loss_identity.item(), loss_cls_recov.item(), loss_D_A.item(), loss_D_B.item()
                    , 100.*correct_real / total, loss_cls_clear.item(), 100.*correct_recov / total, loss_cls_recov.item(), 100.*correct_target / total, loss_cls_fake.item()))

            print('Train, {}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}, {:.4f}, {:.2f}, {:.4f}, {:.2f}, {:.4f}'
                  .format(epoch, it, loss_GAN.item(), loss_cycle.item(), loss_identity.item(), loss_cls_recov.item(), loss_D_A.item(), loss_D_B.item()
                    , 100.*correct_real / total, loss_cls_clear.item(), 100.*correct_recov / total, loss_cls_recov.item(), 100.*correct_target / total, loss_cls_fake.item()))

    utils.print_log('')

def test(state_info, Source_test_loader, Target_test_loader, criterion, realA_sample, realB_sample, epoch):
    
    utils.print_log('Type, Epoch, Batch, accSource, accTarget')
    state_info.set_test_mode()
    correct_src = torch.tensor(0, dtype=torch.float32)
    correct_target = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)
    total_loss_src = 0
    total_loss_target = 0

    if realA_sample.size(1) == 1:
        realA_sample = torch.cat([realA_sample, realA_sample, realA_sample], 1)

    if realB_sample.size(1) == 1:
        realB_sample = torch.cat([realB_sample, realB_sample, realB_sample], 1)    

    for it, ((real_A, Source_y), (real_B, Target_y)) in enumerate(zip(Source_test_loader, Target_test_loader)):

        if real_B.size(0) != real_A.size(0):
            continue

        if real_A.size(1) == 1:
            real_A = torch.cat([real_A, real_A, real_A], 1)

        if real_B.size(1) == 1:
            real_B = torch.cat([real_B, real_B, real_B], 1)
        
        batch_size = real_A.size(0)

        real_A, Source_y = to_var(real_A, FloatTensor), to_var(Source_y, LongTensor)
        real_B, Target_y = to_var(real_B, FloatTensor), to_var(Target_y, LongTensor)

        
        output_cls_src = state_info.cls_src(real_A) # Classifier
        # loss_cls_src = criterion(output_cls_src, Source_y)        

        output_cls_target = state_info.cls_target(real_B) # Classifier
        # loss_cls_target = criterion(output_cls_target, Target_y)

        total += float(batch_size)
        _, predicted_src = torch.max(output_cls_src.data, 1)
        correct_src += float(predicted_src.eq(Source_y.data).cpu().sum())

        _, predicted_target = torch.max(output_cls_target.data, 1)
        correct_target += float(predicted_target.eq(Target_y.data).cpu().sum())

    make_sample_image(state_info, epoch, realA_sample, realB_sample) # img_gen_src, Source_y, img_gen_target, Target_y

    source_prediction_max_result.append(correct_src)
    target_prediction_max_result.append(correct_target)

    utils.print_log('Test, {}, {}, {:.2f}, {:.2f}'.format(epoch, it, 100.*correct_src / total, 100.*correct_target / total))
    print('Test, {}, {}, {:.2f}, {:.2f}'.format(epoch, it, 100.*correct_src / total, 100.*correct_target / total))

    return 100.*correct_target / total


def make_sample_image(state_info, epoch, realA_sample, realB_sample):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    img_path1 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/src'))
    img_path2 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/target'))
    img_path3 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/x'))
    img_path4 = utils.make_directory(os.path.join(utils.default_model_dir, 'images/e'))

    z = Variable(FloatTensor(np.random.normal(0, 1, (realA_sample.size(0), args.latent_dim))))

    fake_B, x, e = state_info.G_AB(realA_sample, z)
    fake_A = state_info.G_BA(realB_sample)

    realA, fake_B, x, e = to_data(realA_sample), to_data(fake_B), to_data(x), to_data(e)
    realB, fake_A = to_data(realB_sample), to_data(fake_A)

    makeAtoB = merge_images(realA_sample, fake_B)
    makeX = merge_images(realA_sample, x)
    makeE = merge_images(realA_sample, e)
    makeBtoA = merge_images(realB_sample, fake_A)

    save_image(makeAtoB.data, os.path.join(img_path1, '%d.png' % epoch), normalize=True)
    save_image(makeBtoA.data, os.path.join(img_path2, '%d.png' % epoch), normalize=True)
    save_image(makeX.data, os.path.join(img_path3, '%d.png' % epoch), normalize=True)
    save_image(makeE.data, os.path.join(img_path4, '%d.png' % epoch), normalize=True)

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