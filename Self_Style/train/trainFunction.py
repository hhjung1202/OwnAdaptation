import torch
from torch.autograd import Variable
import utils
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os

def to_var(x, dtype):
    return Variable(x.type(dtype))

def train(args, state_info, Train_loader, Test_loader, epoch):
    cuda = True if torch.cuda.is_available() else False
    device = 'cuda' if cuda else 'cpu'
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    criterion = torch.nn.CrossEntropyLoss().to(device)

    train_Size = torch.tensor(0, dtype=torch.float32)
    correct_Real = torch.tensor(0, dtype=torch.float32)
    correct_Real2 = torch.tensor(0, dtype=torch.float32)

    # train
    state_info.model.train()

    utils.print_log('Type, Epoch, Batch, loss')
    tot_loss = 0
    for it, (x, y, x_rot, y_rot) in enumerate(Train_loader):

        x, y, x_rot, y_rot = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(x_rot, FloatTensor), to_var(y_rot, LongTensor)
        state_info.optim_model.zero_grad()

        if args.type == "self":
            rot_cls, logits = state_info.forward(x_rot)
            loss = criterion(rot_cls, y_rot)
        elif args.type == "cls":
            rot_cls, logits = state_info.forward(x)
            loss = criterion(logits, y)

        loss.backward()
        state_info.optim_model.step()
        tot_loss += loss.item()

    utils.print_log('Train, {}, {}, {:.6f}'.format(epoch, it, tot_loss))
    print('Train, {}, {}, {:.6f}'.format(epoch, it, tot_loss))

    epoch_result = test(args, state_info, Test_loader, epoch)
    return epoch_result


def test(args, state_info, Test_loader, epoch):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    
    testSize = torch.tensor(0, dtype=torch.float32)
    correct_Test = torch.tensor(0, dtype=torch.float32)

    # test
    state_info.model.eval()
    for it, (x, y, x_rot, y_rot) in enumerate(Test_loader):

        x, y, x_rot, y_rot = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(x_rot, FloatTensor), to_var(y_rot, LongTensor)
        rot_cls, logits = state_info.forward(x_rot)

        if args.type == "self":
            rot_cls, logits = state_info.forward(x_rot)
            _, pred = torch.max(rot_cls.softmax(dim=1), 1)
            correct_Test += float(pred.eq(y_rot.data).cpu().sum())
            testSize += float(x_rot.size(0))

        elif args.type == "cls":
            rot_cls, logits = state_info.forward(x)
            _, pred = torch.max(logits.softmax(dim=1), 1)
            correct_Test += float(pred.eq(y.data).cpu().sum())
            testSize += float(x.size(0))

    utils.print_log('Type, Epoch, Batch, Percentage')
    utils.print_log('Test, {}, {}, {:.3f}'.format(epoch, it, 100.*correct_Test / testSize))
    print('Test, {}, {}, {:.3f}'.format(epoch, it, 100.*correct_Test / testSize))

    return 100.*correct_Test / testSize