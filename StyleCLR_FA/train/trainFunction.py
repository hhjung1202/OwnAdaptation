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
    for it in range(args.iteration):
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
        loss_a, loss_c, loss_s = state_info.forward(x)
        total_loss = 0
        if args.loss[0] is 1: total_loss += loss_a;
        if args.loss[1] is 1: total_loss += loss_c;
        if args.loss[2] is 1: total_loss += loss_s;

        total_loss.backward()
        state_info.optim_model.step()

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(epoch, it, total_loss.item(), loss_a.item()
                , loss_c.item(), loss_s.item()))
            print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(epoch, it, total_loss.item(), loss_a.item()
                , loss_c.item(), loss_s.item()))

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
        # y_style_ = y.view(-1,1).repeat(1,args.n).view(-1)
        
        content, style, recon, adain = state_info.test(x)

        # _, pred = torch.max(out.softmax(dim=1), 1)
        # correct_Real += float(pred.eq(y.data).cpu().sum())

        # _, pred = torch.max(out_style.softmax(dim=1), 1)
        # correct_Real2 += float(pred.eq(y_style_.data).cpu().sum()) // args.n

        # testSize += float(x.size(0))
        make_sample_image(content, style, recon, adain, epoch)

        break

    utils.print_log('Type, Epoch, Batch, Percentage')

    utils.print_log('Test, {}, {}, {:.3f}, {:.3f}'
          .format(epoch, it, 100.*correct_Real / testSize, 100.*correct_Real2 / testSize))
    print('Test, {}, {}, {:.3f}, {:.3f}'
          .format(epoch, it, 100.*correct_Real / testSize, 100.*correct_Real2 / testSize))

    return (100.*correct_Real / testSize, 100.*correct_Real2 / testSize)

def make_sample_image(content, style, recon, adain, epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    img_path = utils.make_directory(os.path.join(utils.default_model_dir, 'images/'))
    content, style, recon, adain = to_data(content), to_data(style), to_data(recon), to_data(adain),
    merged = merge_images(content, style, recon, adain)
    save_image(merged.data, os.path.join(img_path, '%d.png' % epoch), normalize=True)

def merge_images(content, style, recon, adain, row=10):
    _, _, h, w = content.shape
    merged = np.zeros([3, row*h, row*w*4])
    for idx, (c, s, r, a) in enumerate(zip(content, style, recon, adain)):
        i = idx // row
        j = idx % row
        if i is row:
            break
        merged[:, i*h:(i+1)*h, (j*4)*h:(j*4+1)*h] = c
        merged[:, i*h:(i+1)*h, (j*4+1)*h:(j*4+2)*h] = s
        merged[:, i*h:(i+1)*h, (j*4+2)*h:(j*4+3)*h] = r
        merged[:, i*h:(i+1)*h, (j*4+3)*h:(j*4+4)*h] = a

    return torch.from_numpy(merged)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()
