import torch
from torch.autograd import Variable
import utils
from .pseudoDataset import *

def to_var(x, dtype):
    return Variable(x.type(dtype))

def train_step1(state_info, Train_loader, Test_loader, Memory, criterion, epoch):
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
        out, z = state_info.forward(x)
        Memory.Batch_Insert(z, y)
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

    epoch_result = test(state_info, Test_loader, epoch)
    return epoch_result


def train_step2(args, state_info, Train_loader, Test_loader, Memory, criterion, epoch, AnchorSet):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    correct_Noise = torch.tensor(0, dtype=torch.float32)
    correct_Pseudo = torch.tensor(0, dtype=torch.float32)
    correct_Real = torch.tensor(0, dtype=torch.float32)
    train_Size = torch.tensor(0, dtype=torch.float32)
    Pseudo_Noise = torch.tensor(0, dtype=torch.float32)
    Pseudo_Real = torch.tensor(0, dtype=torch.float32)

    Anchor_Image, Anchor_label = AnchorSet
    Anchor_Image, Anchor_label = to_var(Anchor_Image, FloatTensor), to_var(Anchor_label, LongTensor)

    # train
    state_info.model.train()
    utils.print_log('Type, Epoch, Batch, total, Noise_Cls, Pseudo_Cls, Reg_Noise, Reg_Pseudo, Model_Real%, Pseu_Real%')
    for it, (x, y, label) in enumerate(Train_loader):


        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)

        state_info.optim_model.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
        out, z = state_info.forward(x)
        _, model_pred = torch.max(out.data, 1)
        Memory.Batch_Insert(z, model_pred)

        # ------------------------------------------------------------
        _, Anchor_z = state_info.forward(Anchor_Image)
        Memory.Anchor_Insert(Anchor_z, Anchor_label)

        pseudo_label = Memory.Calc_Pseudolabel(z)
        reg_N = Memory.get_Regularizer(z, y, reduction='mean')
        reg_P = Memory.get_Regularizer(z, pseudo_label, reduction='mean')

        # Anchor_Image args.Anchor * 10, CH, W, H
        # ------------------------------------------------------------
        loss_N = criterion(out, y)
        loss_P = criterion(out, pseudo_label)

        total = args.t0 * loss_N + args.t1 * loss_P + args.t2 * reg_N + args.t3 * reg_P

        total.backward()
        state_info.optim_model.step()

        Pseudo_Real += float(pseudo_label.eq(label).sum())
        correct_Real += float(model_pred.eq(label.data).cpu().sum())
        train_Size += float(x.size(0))

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'
                  .format(epoch, it, total.item(), loss_N.item(), loss_P.item(), reg_N.item(), reg_P.item()
                    , 100.*correct_Real / train_Size
                    , 100.*Pseudo_Real / train_Size))
            print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'
                  .format(epoch, it, total.item(), loss_N.item(), loss_P.item(), reg_N.item(), reg_P.item()
                    , 100.*correct_Real / train_Size
                    , 100.*Pseudo_Real / train_Size))

    epoch_result = test(state_info, Test_loader, epoch)
    return epoch_result

def train_step3(args, state_info, Train_loader, Test_loader, Memory, criterion, epoch, AnchorSet):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    correct_Real = torch.tensor(0, dtype=torch.float32)
    Pseudo_Real = torch.tensor(0, dtype=torch.float32)

    Anchor_Image, Anchor_label = AnchorSet
    Anchor_Image, Anchor_label = to_var(Anchor_Image, FloatTensor), to_var(Anchor_label, LongTensor)

    # train
    state_info.model.train()
    utils.print_log('Type, Epoch, Batch, total, Noise_Cls, Pseudo_Cls, Reg_Noise, Reg_Pseudo, Model_Real%, Pseu_Real%')
    for it, (x, y, label) in enumerate(Train_loader):


        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)

        state_info.optim_model.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
        out, z = state_info.forward(x)
        _, model_pred = torch.max(out.data, 1)
        Memory.Batch_Insert(z, model_pred)

        # ------------------------------------------------------------
        _, Anchor_z = state_info.forward(Anchor_Image)
        Memory.Anchor_Insert(Anchor_z, Anchor_label)

        pseudo_label = Memory.Calc_Pseudolabel(z)
        reg_P = Memory.get_Regularizer(z, pseudo_label, reduction='mean')
        reg_N = Memory.get_Regularizer(z, y, reduction='mean')

        # Anchor_Image args.Anchor * 10, CH, W, H
        # ------------------------------------------------------------
        loss_N = criterion(out, y)
        loss_P = criterion(out, pseudo_label)

        total = args.t4 * loss_N + args.t5 * loss_P + args.t6 * reg_N + args.t7 * reg_P

        total.backward()
        state_info.optim_model.step()

        Pseudo_Real += float(pseudo_label.eq(label).sum())
        correct_Real += float(model_pred.eq(label.data).cpu().sum())
        train_Size += float(x.size(0))

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'
                  .format(epoch, it, total.item(), loss_N.item(), loss_P.item(), reg_N.item(), reg_P.item()
                    , 100.*correct_Real / train_Size
                    , 100.*Pseudo_Real / train_Size))
            print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'
                  .format(epoch, it, total.item(), loss_N.item(), loss_P.item(), reg_N.item(), reg_P.item()
                    , 100.*correct_Real / train_Size
                    , 100.*Pseudo_Real / train_Size))

    epoch_result = test(state_info, Test_loader, epoch)
    return epoch_result


def test(state_info, Test_loader, epoch):
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

        out, z = state_info.forward(x)

        _, pred = torch.max(out.data, 1)
        correct_Test += float(pred.eq(y.data).cpu().sum())
        testSize += float(x.size(0))

    utils.print_log('Type, Epoch, Batch, Percentage')

    utils.print_log('Test, {}, {}, {:.3f}'
          .format(epoch, it, 100.*correct_Test / testSize))
    print('Test, {}, {}, {:.3f}'
          .format(epoch, it, 100.*correct_Test / testSize))

    return 100.*correct_Test / testSize