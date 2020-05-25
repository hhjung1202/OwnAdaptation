import torch
from torch.autograd import Variable
import utils
import torch.nn.functional as F

def to_var(x, dtype):
    return Variable(x.type(dtype))

def train(args, state_info, Train_loader, Test_loader, criterion, epoch):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    train_Size = torch.tensor(0, dtype=torch.float32)
    correct_Real = torch.tensor(0, dtype=torch.float32)
    correct_Real2 = torch.tensor(0, dtype=torch.float32)

    # train
    state_info.model.train()
    utils.print_log('Type, Epoch, Batch, total, Noise_Cls, Pseudo_Cls, Reg_Noise, Reg_Pseudo, Model_Real%, Pseu_Real%')
    for it, (x, y, label) in enumerate(Train_loader):

        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
        state_info.optim_model.zero_grad()

        out_IN = state_info.forward_IN(x)
        out_BN = state_info.forward_BN(x)

        loss_IN = criterion(out_IN, label)
        loss_BN = criterion(out_BN, label)
        total = args.weight[0] * loss_BN + args.weight[1] * loss_IN
        total.backward()
        state_info.optim_model.step()

        _, pred = torch.max(out_BN.softmax(dim=1), 1)
        correct_Real += float(pred.eq(label.data).cpu().sum())

        _, pred = torch.max(out_IN.softmax(dim=1), 1)
        correct_Real2 += float(pred.eq(label.data).cpu().sum())

        train_Size += float(x.size(0))

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'.format(epoch, it, total.item(), loss_BN.item(), loss_IN.item()
                , 100.*correct_Real / train_Size, 100.*correct_Real2 / train_Size))
            print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'.format(epoch, it, total.item(), loss_BN.item(), loss_IN.item()
                , 100.*correct_Real / train_Size, 100.*correct_Real2 / train_Size))

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

        out_IN = state_info.forward_IN(x)
        out_BN = state_info.forward_BN(x)

        _, pred = torch.max(out_BN.softmax(dim=1), 1)
        correct_Real += float(pred.eq(y.data).cpu().sum())

        _, pred = torch.max(out_IN.softmax(dim=1), 1)
        correct_Real2 += float(pred.eq(y.data).cpu().sum())

        # _, pred = torch.max(out.data, 1)
        # correct_Test += float(pred.eq(y.data).cpu().sum())
        testSize += float(x.size(0))

    utils.print_log('Type, Epoch, Batch, Percentage')

    utils.print_log('Test, {}, {}, {:.3f}, {:.3f}'
          .format(epoch, it, 100.*correct_Real / testSize, 100.*correct_Real2 / testSize))
    print('Test, {}, {}, {:.3f}, {:.3f}'
          .format(epoch, it, 100.*correct_Real / testSize, 100.*correct_Real2 / testSize))

    return 100.*correct_Test / testSize