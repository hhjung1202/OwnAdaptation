import torch
from torch.autograd import Variable
import utils
from .pseudoDataset import *
import torch.nn.functional as F
from typing import List, Mapping, Optional

def to_var(x, dtype):
    return Variable(x.type(dtype))

def WeightedGradientGamma(weight, low=-1, high=1):
    return weight * (high-low) + low




def hard_label_cross_entropy(input, target, eps=1e-5):
    # input (N, C)
    # target (N) with hard class label
    log_likelihood = input.log_softmax(dim=1)
    nll_loss = F.nll_loss(log_likelihood, target)
    return nll_loss

def soft_label_cross_entropy(input, target, eps=1e-5):
    # input (N, C)
    # target (N, C) with soft label
    log_likelihood = input.log_softmax(dim=1)
    soft_log_likelihood = target * log_likelihood
    nll_loss = -torch.sum(soft_log_likelihood.mean(dim=0))
    return nll_loss

# def Reverse_soft_label_cross_entropy(input, target, eps=1e-5):
#     # input (N, C)
#     # target (N, C) with soft label
#     log_likelihood_reverse = torch.log(1 - input.softmax(dim=1) + eps)
#     soft_log_likelihood = target * log_likelihood_reverse
#     nll_loss = -torch.sum(soft_log_likelihood.mean(dim=0))
#     return nll_loss

def Reverse_hard_label_cross_entropy(input, target, eps=1e-5):
    # input (N, C)
    # target (N) with hard class label
    log_likelihood_reverse = torch.log(1 - input.softmax(dim=1) + eps)
    nll_loss = F.nll_loss(log_likelihood_reverse, target)
    return nll_loss

def Maximize_Pseudo_Entropy_loss(pseudo_soft_label, eps=1e-5):
    loss_Ent = torch.mean(torch.sum(pseudo_soft_label * (torch.log(pseudo_soft_label + eps)), 1))
    return loss_Ent

def soft_label_cross_entropy_diff(input, target, reverse_one, eps=1e-5):
    # input (N, C)
    # target (N, C) with soft label
    log_likelihood = input.log_softmax(dim=1)
    soft_log_likelihood = target * log_likelihood * reverse_one
    nll_loss = -torch.sum(soft_log_likelihood.mean(dim=0))
    return nll_loss

def hard_label_cross_entropy_same(input, target, one, eps=1e-5):
    # input (N, C)
    # target (N) with hard class label
    log_likelihood = input.log_softmax(dim=1) * one
    nll_loss = F.nll_loss(log_likelihood, target)
    return nll_loss

# def posteriori_cross_entropy(input, target, ramda=0.5, eps=1e-5):
#     # input (N, C)
#     # target (N, C) with soft label
#     log_likelihood = input.log_softmax(dim=1)
#     soft_log_likelihood = target * log_likelihood * reverse_one
#     nll_loss = -torch.sum(soft_log_likelihood.mean(dim=0))
#     return nll_loss

# def posteriori_cross_entropy(input, target, gamma=0.5, eps=1e-5):
#     # input (N, C)
#     # target (N) with hard class label
#     posteriori = input.softmax(dim=1)[range(target.size(0)), target] # posteriori (N) 
#     log_likelihood = input.log_softmax(dim=1)[range(target.size(0)), target] # log_likelihood about target (N) 
#     # p > 0.5 정상 학습
#     posteriori[(posteriori < gamma).nonzero().view(-1)] = 0.
#     # p < 0.5 다른 방법 강구
#     soft_log_likelihood = posteriori * log_likelihood
#     nll_loss = -soft_log_likelihood.mean(dim=0)
#     return nll_loss
    

def train_step1(args, state_info, Train_loader, Test_loader, Memory, criterion, epoch):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    state_info.model.train()
    utils.print_log('Type, Epoch, Batch, total, percentage')
    correct = torch.tensor(0, dtype=torch.float32)
    total_Size = torch.tensor(0, dtype=torch.float32)

    for it, (x, y, label) in enumerate(Train_loader):
        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)

        state_info.optim_model.zero_grad()
        out, z = state_info.forward(args, x)
        Memory.Batch_Insert(z, y, torch.ones(z.size(0)).cuda())
        loss = criterion(out, y)
        loss.backward(retain_graph=True)
        state_info.optim_model.step()

        _, pred = torch.max(out.data, 1)
        correct += float(pred.eq(y.data).cpu().sum())
        total_Size += float(x.size(0))
        
        if it % 10 == 0:
            utils.print_log('Init, {}, {}, {:.6f}, {:.3f}'
                  .format(epoch, it, loss.item(), 100.*correct / total_Size))
            print('Init, {}, {}, {:.6f}, {:.3f}'
                  .format(epoch, it, loss.item(), 100.*correct / total_Size))

    epoch_result = test(args, state_info, Test_loader, epoch)
    return epoch_result

def train_step4(args, state_info, Train_loader, Test_loader, Memory, criterion, epoch):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    train_Size = torch.tensor(0, dtype=torch.float32)
    correct_Real = torch.tensor(0, dtype=torch.float32)
    Pseudo_Real = torch.tensor(0, dtype=torch.float32)

    # train
    state_info.model.train()
    utils.print_log('Type, Epoch, Batch, total, Noise_Cls, Pseudo_Cls, Reg_Noise, Reg_Pseudo, Model_Real%, Pseu_Real%')
    for it, (x, y, label) in enumerate(Train_loader):

        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
        y_one = torch.cuda.FloatTensor(y.size(0), 10).zero_().scatter_(1, y.view(-1, 1), 1)

        state_info.optim_model.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
        out, z = state_info.forward(args, x)
        model_pred, model_y = torch.max(out.softmax(dim=1), 1)
        Memory.Batch_Insert(z, model_y, model_pred)

        # ------------------------------------------------------------

        memory_soft_label = Memory.Calc_Pseudolabel(z)
        posteriori = out.softmax(dim=1)[range(out.size(0)), y] # posteriori (N) 
        print(posteriori.size())
        print(y_one.size())
        print(memory_soft_label.size())
        label = posteriori * y_one + (1-posteriori) * memory_soft_label
        label = label.pow(1/args.T) / label.pow(1/args.T).sum() # Sharpening

        # if args.grad == "T":
        loss_P_soft = soft_label_cross_entropy(out, label)
        loss_Ent = Maximize_Pseudo_Entropy_loss(memory_soft_label)
        reg_P = Memory.get_Regularizer(z, torch.max(label, 1)[1], reduction='mean')

        total = args.weight[0] * loss_P_soft + args.weight[1] * loss_Ent + args.weight[2] * reg_P

        total.backward()
        state_info.optim_model.step()

        Pseudo_Real += float(pseudo_hard_label.eq(label).sum())
        correct_Real += float(model_y.eq(label.data).cpu().sum())
        train_Size += float(x.size(0))

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'
                  .format(epoch, it, total.item(), loss_P_hard.item(), loss_P_soft.item(), reg_P.item(), loss_Ent.item()
                    , 100.*correct_Real / train_Size
                    , 100.*Pseudo_Real / train_Size))
            print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'
                  .format(epoch, it, total.item(), loss_P_hard.item(), loss_P_soft.item(), reg_P.item(), loss_Ent.item()
                    , 100.*correct_Real / train_Size
                    , 100.*Pseudo_Real / train_Size))

    epoch_result = test(args, state_info, Test_loader, epoch)
    return epoch_result



def test(args, state_info, Test_loader, epoch):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    
    testSize = torch.tensor(0, dtype=torch.float32)
    Similarity_Scale = torch.tensor(0, dtype=torch.float32)
    Similarity_Vector = torch.tensor(0, dtype=torch.float32)
    correct_Test = torch.tensor(0, dtype=torch.float32)

    # test
    state_info.model.eval()
    for it, (x, y) in enumerate(Test_loader):

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)

        out, z = state_info.forward(args, x)

        _, pred = torch.max(out.data, 1)
        correct_Test += float(pred.eq(y.data).cpu().sum())
        testSize += float(x.size(0))

    utils.print_log('Type, Epoch, Batch, Percentage')

    utils.print_log('Test, {}, {}, {:.3f}'
          .format(epoch, it, 100.*correct_Test / testSize))
    print('Test, {}, {}, {:.3f}'
          .format(epoch, it, 100.*correct_Test / testSize))

    return 100.*correct_Test / testSize