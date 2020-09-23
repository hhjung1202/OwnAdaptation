import numpy as np
import os

import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import scipy.ndimage

from augment.cutout import Cutout
from augment.autoaugment_extra import CIFAR10Policy
from utils import load_state_dict

epoch = 300
batch_size = 128

tensorboard_log_dir = './produce_log_resnet50pre_360_10_epoch300'
os.environ['CUDA_VISIBLE_DEVICES']='3'

# CutOut parameters
n_holes = 1
cutout_size = 16
# Rotation parameters
K = 4  # number of rotation(1 ~ 4)
each_rotation_degree = 90

def save_state_dict(model, path):
    torch.save(model.state_dict(), path)

def weights_init_normal(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def rotate_90(img, n):
    return np.rot90(img, n, (1, 2))

def rotate(img, degree):
    ori = img.shape
    rot_img = scipy.ndimage.rotate(img, degree, axes=(-2, -1), mode='nearest',)
    return np.resize(rot_img, ori)

def collate(batch):
    rot_imgs = []
    rot_labels = []
    imgs = []
    labels = []
    for x_aug, label in batch:
        for n in range(K):
            rot_imgs.append(torch.FloatTensor(rotate(x_aug.numpy(), n*each_rotation_degree).copy()))
            rot_labels.append(torch.tensor(n))
        imgs.append(x_aug)
        labels.append(torch.tensor(label))

    return [torch.stack(imgs), torch.stack(labels), torch.stack(rot_imgs), torch.stack(rot_labels)]

# transformers
transform_strongaug = transforms.Compose([
    transforms.ToTensor(),
    Cutout(n_holes=n_holes, length=cutout_size),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    # CIFAR10Policy(),
    transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])

def eval_model(model, test_loader, device, criterion):
    correct = 0
    tot_loss = 0
    num_test = len(test_loader.dataset) * K


    model.eval()
    for img, label, data, label in test_loader:
        with torch.no_grad():
            data = data.to(device)
            label = label.to(device)

            pred = model(data)
            loss = criterion(pred, label)
            tot_loss += loss.item()
            pred_cls = pred.max(1)[1]
            correct += pred_cls.cpu().data.eq(label.cpu().data).sum()

    acc = float(correct) / num_test

    return tot_loss, acc

def train(model, criterion, device, train_loader, test_loader, epoch=100, save_best=False):
    optimizer = torch.optim.Adam(model.parameters(), 2e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=2e-6,
                                                           last_epoch=-1)
    # writer = SummaryWriter(tensorboard_log_dir)

    best_acc = 0.

    print('start training...')
    for epoch in range(epoch):
        tot_loss = 0
        model.train()
        for step, (img, label, rot_img, rot_label) in enumerate(train_loader):
            data = rot_img.to(device)
            label = rot_label.to(device)

            pred = model(data)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('epoch {}, train loss {:.4}'.format(epoch, tot_loss))
        # writer.add_scalar('train loss', loss, epoch)
        # writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        scheduler.step()
        eval_loss, acc = eval_model(model, test_loader, device, criterion)
        print('epoch {}, test loss {:.4}, acc {:.4}'.format(epoch, eval_loss, acc))
        # writer.add_scalar('test/accuracy', acc, epoch)
        # writer.add_scalar('test/loss', loss, epoch)

        if acc > best_acc:
            print('best {:.4} to {:.4}'.format(best_acc, acc))
            best_acc = acc
            if save_best:
                print('save...')
                save_state_dict(model, './rot_cp/rot_best_fe_norm{:.4}'.format(acc))
    print('best is {:.4%}'.format(best_acc))

from models.resnet import ResNet34

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # models
    backbone = ResNet34(10, True)
    num_ftrs = backbone.fc_in
    rot_cls = nn.Sequential(nn.Flatten(), nn.Linear(num_ftrs, K))
    rot_cls.apply(weights_init_normal)

    model = nn.Sequential(backbone, rot_cls).to(device)
    CE = nn.CrossEntropyLoss().to(device)

    # loaders
    trainset = torchvision.datasets.CIFAR10(root='/disk1/CIFAR10', train=True, download=True, transform=transform_strongaug)
    testset = torchvision.datasets.CIFAR10(root='/disk1/CIFAR10', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=2)


    train(model, CE, device, trainloader, testloader, epoch=epoch, save_best=True)
    # load_state_dict('./rot_cp/rot_best_fe0.9445', model)
    # eval_loss, acc = eval_model(nn.Sequential(backbone, rot_cls), testloader, device,nn.CrossEntropyLoss())
    # print('loaded test loss {:.4}, acc {:.4}'.format(eval_loss, acc))



if __name__ == '__main__':
    main()