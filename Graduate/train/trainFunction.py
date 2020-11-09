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

    train_Size = torch.tensor(0, dtype=torch.float32)
    correct = torch.tensor(0, dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # train
    state_info.model.train()

    utils.print_log('Type, Epoch, loss, Acc')
    tot_loss = 0
    
    for it, (x, y) in enumerate(Train_loader):

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)
        state_info.optim_model.zero_grad()

        output = state_info.forward(x)

        loss = criterion(output, y)
        tot_loss += loss
        _, pred = torch.max(output.data, 1)
        correct += pred.eq(y.data).cpu().sum()
        train_Size += x.size(0)
            
        loss.backward()
        state_info.optim_model.step()

    utils.print_log('Train, {}, {:.6f}, {:.4f}'.format(epoch, tot_loss, 100.*correct/train_Size))
    print('Train, {}, {:.6f}, {:.4f}'.format(epoch, tot_loss, 100.*correct/train_Size))

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
    for it, (x, y) in enumerate(Test_loader):

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)
        output = state_info.forward(x)

        _, pred = torch.max(output, 1)
        correct_Test += float(pred.eq(y.data).cpu().sum())
        testSize += float(x.size(0))

    utils.print_log('Type, Epoch, Percentage')
    utils.print_log('Test, {}, {:.3f}'.format(epoch, 100.*correct_Test / testSize))
    print('Test, {}, {:.3f}'.format(epoch, 100.*correct_Test / testSize))

    return 100.*correct_Test / testSize