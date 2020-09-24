import torch
from torch.autograd import Variable
import utils
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os

softmin = torch.nn.Softmin(dim=-1)

def to_var(x, dtype):
    return Variable(x.type(dtype))

def softmin_ce(input, target, weight=1e-1): # y * log(p), p = softmax(-out)
    likelihood = softmin(input * weight)
    print(softmin(input * 1e-5).var(dim=1).mean())
    print(softmin(input * 1e-4).var(dim=1).mean())
    print(softmin(input * 1e-3).var(dim=1).mean())
    print(softmin(input * 1e-2).var(dim=1).mean())
    print(softmin(input * 1e-1).var(dim=1).mean())
    print(softmin(input).var(dim=1).mean())
    log_likelihood = likelihood.log()
    nll_loss = F.nll_loss(log_likelihood, target)
    return nll_loss

def softmax_ce_rev(input, target, weight=1e-1): # y * log(1-p), p = softmax(out)
    likelihood = F.softmax(input * weight, dim=-1)
    print(F.softmax(input * 1e-5, dim=-1).var(dim=1).mean())
    print(F.softmax(input * 1e-4, dim=-1).var(dim=1).mean())
    print(F.softmax(input * 1e-3, dim=-1).var(dim=1).mean())
    print(F.softmax(input * 1e-2, dim=-1).var(dim=1).mean())
    print(F.softmax(input * 1e-1, dim=-1).var(dim=1).mean())
    print(F.softmax(input, dim=-1).var(dim=1).mean())
    log_likelihood_reverse = torch.log(1 - likelihood)
    nll_loss = F.nll_loss(log_likelihood_reverse, target)
    return nll_loss

def train(args, state_info, Train_loader, Test_loader, epoch):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    train_Size = torch.tensor(0, dtype=torch.float32)
    correct_Real = torch.tensor(0, dtype=torch.float32)
    correct_Real2 = torch.tensor(0, dtype=torch.float32)

    # train
    state_info.model.train()

    utils.print_log('Type, Epoch, Batch, total, Noise_Cls, Pseudo_Cls, Reg_Noise, Reg_Pseudo, Model_Real%, Pseu_Real%')
    
    for it, (x, y) in enumerate(Train_loader):

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)
        state_info.optim_model.zero_grad()

        logits, st_mse, st_label = state_info.forward(x)

        # style_loss = softmin_ce(st_mse, st_label, weight=1e-1)
        # style_loss = softmax_ce_rev(st_mse, st_label, weight=1e-1)

        if args.loss[0] is 0: style_loss = softmin_ce(st_mse, st_label, weight=1e-1);
        if args.loss[0] is 1: style_loss = softmax_ce_rev(st_mse, st_label, weight=1e-1);

        style_loss.backward()
        state_info.optim_model.step()

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.9f}'.format(epoch, it, style_loss.item()))
            print('Train, {}, {}, {:.9f}'.format(epoch, it, style_loss.item()))

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
        logits, st_mse, st_label = state_info.forward(x)

        _, pred = torch.min(st_mse, 1)
        correct_Test += float(pred.eq(st_label.data).cpu().sum())

        testSize += float(x.size(0))

    utils.print_log('Type, Epoch, Batch, Percentage')
    utils.print_log('Test, {}, {}, {:.3f}'.format(epoch, it, 100.*correct_Test / testSize))
    print('Test, {}, {}, {:.3f}'.format(epoch, it, 100.*correct_Test / testSize))

    return 100.*correct_Test / testSize