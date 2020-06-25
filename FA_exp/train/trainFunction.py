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

def mean_cross_entropy(input, input2, target):
    # input (N, C)
    # target (N, C) with soft label
    log_likelihood = torch.log((input.softmax(dim=1) + input2.softmax(dim=1))/2)
    nll_loss = F.nll_loss(log_likelihood, target)
    return nll_loss

def train(args, state_info, Train_loader, Test_loader, criterion, epoch):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    train_Size = torch.tensor(0, dtype=torch.float32)
    correct_Real = torch.tensor(0, dtype=torch.float32)

    # train
    state_info.model.train()
    utils.print_log('Type, Epoch, Batch, total, Noise_Cls, Pseudo_Cls, Reg_Noise, Reg_Pseudo, Model_Real%, Pseu_Real%')
    for it, (x, y, label) in enumerate(Train_loader):

        print(args.fixed_perm)
        perm = torch.randperm(x.size(0)) if args.fixed_perm else None
        print(perm)
        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
        label_one = FloatTensor(y.size(0), 10).zero_().scatter_(1, y.view(-1, 1), 1)
        suffle_label, suffle_label_one = y[perm][0], label_one[perm][0]

        l = np.random.beta(0.75, 0.75)
        l = max(l, 1-l)
        mixed_label =  l * label_one + (1-l) * suffle_label_one

        state_info.optim_model.zero_grad()

        out, origin, style_loss= state_info.forward(x, perm) # content loss, style loss

        loss = {    0: mean_cross_entropy(out, origin, y),
                    1: soft_label_cross_entropy(out, mixed_label),
                    2: criterion(out, suffle_label),
                    3: criterion(out, y)}[args.case]

        if style_loss is None:
            style_loss = FloatTensor([0])
        total = loss + args.weight[0] * style_loss
        total.backward()
        state_info.optim_model.step()

        _, pred = torch.max(out.softmax(dim=1), 1)
        correct_Real += float(pred.eq(label.data).cpu().sum())

        train_Size += float(x.size(0))

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.3f}'.format(epoch, it, total.item(), loss.item(), style_loss.item()
                , 100.*correct_Real / train_Size))
            print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.3f}'.format(epoch, it, total.item(), loss.item(), style_loss.item()
                , 100.*correct_Real / train_Size))

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
    correct_Real = torch.tensor(0, dtype=torch.float32)
    correct_Real2 = torch.tensor(0, dtype=torch.float32)

    # test
    state_info.model.eval()
    for it, (x, y) in enumerate(Test_loader):

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)
        
        perm = torch.randperm(x.size(0)) if args.fixed_perm else None
        origin_perm = LongTensor([i for i in range(x.size(0))])
        
        out, origin, _ = state_info.forward(x, perm) # content loss, style loss

        _, pred = torch.max(out.softmax(dim=1), 1)
        correct_Real += float(pred.eq(y.data).cpu().sum())

        _, pred = torch.max(origin.softmax(dim=1), 1)
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