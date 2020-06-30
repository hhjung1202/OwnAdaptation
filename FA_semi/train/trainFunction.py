import torch
from torch.autograd import Variable
import utils
import torch.nn.functional as F
import numpy as np

def to_var(x, dtype):
    return Variable(x.type(dtype))

def soft_label_cross_entropy(input, target):
    # input (N, C)
    # target (N, C) with soft label
    log_likelihood = input.log_softmax(dim=1)
    soft_log_likelihood = target * log_likelihood
    nll_loss = -torch.sum(soft_log_likelihood.mean(dim=0))
    return nll_loss

def train(args, state_info, labeled_trainloader, unlabeled_trainloader, test_loader, criterion, epoch):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    train_Size = torch.tensor(0, dtype=torch.float32)
    correct_Real = torch.tensor(0, dtype=torch.float32)
    correct_Real2 = torch.tensor(0, dtype=torch.float32)

    # train
    state_info.model.train()

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    utils.print_log('Type, Epoch, Batch, total, Noise_Cls, Pseudo_Cls, Reg_Noise, Reg_Pseudo, Model_Real%, Pseu_Real%')
    for batch_idx in range(args.iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            inputs_u, _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, _ = unlabeled_train_iter.next()

        if inputs_x.size(0) is not inputs_u.size(0):
            continue

        inputs_x, inputs_u, targets_x = to_var(inputs_x, FloatTensor), to_var(inputs_u, FloatTensor), to_var(targets_x, LongTensor)
        state_info.optim_model.zero_grad()

        out, style_loss = state_info.forward(inputs_x, inputs_u)

        loss = criterion(out, targets_x)
        loss.backward()

        state_info.optim_model.step()

        _, pred = torch.max(out.softmax(dim=1), 1)
        correct_Real += float(pred.eq(targets_x.data).cpu().sum())

        train_Size += float(inputs_x.size(0))

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.3f}'.format(epoch, it, loss.item(), 100.*correct_Real / train_Size))
            print('Train, {}, {}, {:.6f}, {:.3f}'.format(epoch, it, loss.item(), 100.*correct_Real / train_Size))

    epoch_result = test(args, state_info, test_loader, epoch)
    return epoch_result



def test(args, state_info, Test_loader, epoch):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    
    testSize = torch.tensor(0, dtype=torch.float32)
    Similarity_Scale = torch.tensor(0, dtype=torch.float32)
    Similarity_Vector = torch.tensor(0, dtype=torch.float32)
    correct_Test = torch.tensor(0, dtype=torch.float32)
    correct_Real = torch.tensor(0, dtype=torch.float32)
    correct_Real2 = torch.tensor(0, dtype=torch.float32)

    # test
    state_info.model.eval()
    for it, (x, y) in enumerate(Test_loader):

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)
        
        perm = torch.randperm(x.size(0))

        out1 = state_info.forward(x, x)
        out2 = state_info.forward(x, x[perm])

        _, pred = torch.max(out1.softmax(dim=1), 1)
        correct_Real += float(pred.eq(y.data).cpu().sum())

        _, pred = torch.max(out2.softmax(dim=1), 1)
        correct_Real2 += float(pred.eq(y.data).cpu().sum())

        # _, pred = torch.max(out.data, 1)
        # correct_Test += float(pred.eq(y.data).cpu().sum())
        testSize += float(x.size(0))

    utils.print_log('Type, Epoch, Batch, Percentage')

    utils.print_log('Test, {}, {}, {:.3f}, {:.3f}'
          .format(epoch, it, 100.*correct_Real / testSize, 100.*correct_Real2 / testSize))
    print('Test, {}, {}, {:.3f}, {:.3f}'
          .format(epoch, it, 100.*correct_Real / testSize, 100.*correct_Real2 / testSize))

    # return 100.*correct_Test / testSize
    return (100.*correct_Real / testSize, 100.*correct_Real2 / testSize)

