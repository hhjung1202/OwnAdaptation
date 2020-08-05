import torch
from torch.autograd import Variable
import utils
import torch.nn.functional as F
import numpy as np

def to_var(x, dtype):
    return Variable(x.type(dtype))


def train(args, state_info, labeled_trainloader, unlabeled_trainloader, test_loader, epoch):
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
    # for it in range(args.iteration):
    #     try:
    #         inputs_x, targets_x = labeled_train_iter.next()
    #     except:
    #         labeled_train_iter = iter(labeled_trainloader)
    #         inputs_x, targets_x = labeled_train_iter.next()

    #     try:
    #         inputs_u, _ = unlabeled_train_iter.next()
    #     except:
    #         unlabeled_train_iter = iter(unlabeled_trainloader)
    #         inputs_u, _ = unlabeled_train_iter.next()

    #     if inputs_x.size(0) is not inputs_u.size(0):
    #         continue

    #     inputs_x, inputs_u, targets_x = to_var(inputs_x, FloatTensor), to_var(inputs_u, FloatTensor), to_var(targets_x, LongTensor)
    #     state_info.optim_model.zero_grad()

    #     loss_s, JS_loss, loss_u, style_loss, content_loss = state_info.forward(inputs_x, targets_x, inputs_u)

    #     total_loss = 0
    #     if args.loss[0] is 1: total_loss += loss_s;
    #     if args.loss[1] is 1: total_loss += JS_loss;
    #     if args.loss[2] is 1: total_loss += loss_u;
    #     if args.loss[3] is 1: total_loss += style_loss;
    #     if args.loss[4] is 1: total_loss += content_loss;

    #     total_loss.backward()
    #     state_info.optim_model.step()

    #     if it % 10 == 0:
    #         utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(epoch, it, total_loss.item(), loss_s.item()
    #             , JS_loss.item(), loss_u.item(), style_loss.item(), content_loss.item()))
    #         print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(epoch, it, total_loss.item(), loss_s.item()
    #             , JS_loss.item(), loss_u.item(), style_loss.item(), content_loss.item()))

    epoch_result = test(args, state_info, test_loader, epoch)
    return epoch_result



def test(args, state_info, Test_loader, epoch):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    
    testSize = torch.tensor(0, dtype=torch.float32)
    correct_Test = torch.tensor(0, dtype=torch.float32)
    correct_Real = torch.tensor(0, dtype=torch.float32)
    correct_Real2 = torch.tensor(0, dtype=torch.float32)

    # test
    state_info.model.eval()
    for it, (x, y) in enumerate(Test_loader):

        x, y = to_var(x, FloatTensor), to_var(y, LongTensor)
        y_style_ = y.view(-1,1).repeat(1,args.n).view(-1)
        
        out, out_style = state_info.test(x)



        _, pred = torch.max(out.softmax(dim=1), 1)
        correct_Real += float(pred.eq(y.data).cpu().sum())

        _, pred = torch.max(out_style.softmax(dim=1), 1)
        correct_Real2 += float(pred.eq(y_style_.data).cpu().sum()) // args.n

        testSize += float(x.size(0))

    utils.print_log('Type, Epoch, Batch, Percentage')

    utils.print_log('Test, {}, {}, {:.3f}, {:.3f}'
          .format(epoch, it, 100.*correct_Real / testSize, 100.*correct_Real2 / testSize))
    print('Test, {}, {}, {:.3f}, {:.3f}'
          .format(epoch, it, 100.*correct_Real / testSize, 100.*correct_Real2 / testSize))

    return (100.*correct_Real / testSize, 100.*correct_Real2 / testSize)

