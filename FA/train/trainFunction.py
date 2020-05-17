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
    correct_Rot = torch.tensor(0, dtype=torch.float32)
    Pseudo_Real = torch.tensor(0, dtype=torch.float32)

    # train
    state_info.model.train()
    utils.print_log('Type, Epoch, Batch, total, Noise_Cls, Pseudo_Cls, Reg_Noise, Reg_Pseudo, Model_Real%, Pseu_Real%')
    for it, (x, y, label) in enumerate(Train_loader):

        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)

        state_info.optim_model.zero_grad()
        logits, logits_rot, Rot_label = state_info.forward(x)
        Rot_label = to_var(Rot_label, LongTensor)
        loss = criterion(logits, label)
        loss_rot = criterion(logits_rot, Rot_label)
        total = loss + loss_rot
        total.backward()
        state_info.optim_model.step()

        _, pred = torch.max(logits.softmax(dim=1), 1)
        correct_Real += float(pred.eq(label.data).cpu().sum())

        _, pred = torch.max(logits_rot.softmax(dim=1), 1)
        correct_Rot += float(pred.eq(Rot_label.data).cpu().sum())

        train_Size += float(x.size(0))

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'.format(epoch, it, loss.item(), loss_rot.item()
                , 100.*correct_Real / train_Size, 100.*correct_Rot / train_Size))
            print('Train, {}, {}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'.format(epoch, it, loss.item(), loss_rot.item()
                , 100.*correct_Real / train_Size, 100.*correct_Rot / train_Size))

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
    correct_Rot = torch.tensor(0, dtype=torch.float32)

    # test
    state_info.model.eval()
    for it, (x, y) in enumerate(Test_loader):

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)

        logits, logits_rot, Rot_label = state_info.forward(x)

        _, pred = torch.max(logits.data, 1)
        correct_Test += float(pred.eq(y.data).cpu().sum())

        _, pred = torch.max(logits_rot.data, 1)
        correct_Rot += float(pred.eq(Rot_label.data).cpu().sum())

        testSize += float(x.size(0))

    utils.print_log('Type, Epoch, Batch, Percentage')

    utils.print_log('Test, {}, {}, {:.3f}, {:.3f}'
          .format(epoch, it, 100.*correct_Test / testSize, 100.*correct_Rot / testSize))
    print('Test, {}, {}, {:.3f}, {:.3f}'
          .format(epoch, it, 100.*correct_Test / testSize, 100.*correct_Rot / testSize))

    return 100.*correct_Test / testSize