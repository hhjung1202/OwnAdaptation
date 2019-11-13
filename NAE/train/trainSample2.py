import torch
from torch.autograd import Variable
import os
import time
import utils

# lr, batch

def to_var(x, dtype):
    return Variable(x.type(dtype))

def WeightedGradientGamma(weight, low=-1, high=1):
    return weight * (high-low) + low

# def WeightedExponentialGradientGamma(weight, low=-1, high=1):
#     return weight * (high-low) + low

def train_Sample2(args, state_info, Noise_Sample_loader, Noise_Test_loader): # all 

    best_prec_result = torch.tensor(0, dtype=torch.float32)
    start_time = time.time()
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    mode = 'sample'
    utils.default_model_dir = os.path.join(args.dir, mode + "2")

    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    # criterion_GAN = torch.nn.MSELoss()

    start_epoch = 0
    checkpoint = utils.load_checkpoint(utils.default_model_dir)    
    if not checkpoint:
        args.last_epoch = -1
        state_info.learning_scheduler_init(args, mode)
    else:
        start_epoch = checkpoint['epoch'] + 1
        best_prec_result = checkpoint['Best_Prec']
        state_info.load_state_dict(checkpoint, mode)
        args.last_epoch = start_epoch
        state_info.learning_scheduler_init(args, mode)

    utils.print_log('Type, Epoch, Batch, loss, BCE, KLD, CE')

    for epoch in range(start_epoch, args.epoch):

        correct_Noise = torch.tensor(0, dtype=torch.float32)
        correct_Real = torch.tensor(0, dtype=torch.float32)
        correct_Test = torch.tensor(0, dtype=torch.float32)
        total = torch.tensor(0, dtype=torch.float32)

        # train
        state_info.sample.train()
        for it, (Sample, Sy, label_Sy) in enumerate(Noise_Sample_loader):

            Sample, Sy, label_Sy = to_var(Sample, FloatTensor), to_var(Sy, LongTensor), to_var(label_Sy, LongTensor)

            if args.grad == "T":
                weight = label_Sy.eq(Sy).type(FloatTensor).view(-1,1)
                zero = torch.zeros(weight.size()).type(FloatTensor)
                reverse_weight = weight.eq(zero).type(LongTensor).view(-1)
                Gamma = WeightedGradientGamma(weight, low=args.low, high=args.high)

            elif args.grad == "F":
                Gamma = 1

            Sout = state_info.forward_Sample(Sample, gamma=Gamma)

            _, pred = torch.max(Sout.data, 1)
            label = Sy * weight.type(LongTensor).view(-1) + pred * reverse_weight
            state_info.optim_Sample.zero_grad()
            loss = criterion(Sout, label)
            loss.backward()
            state_info.optim_Sample.step()

            # _, pred = torch.max(Sout.data, 1)
            correct_Noise += float(pred.eq(Sy.data).cpu().sum())
            correct_Real += float(pred.eq(label_Sy.data).cpu().sum())
            total += float(Sample.size(0))
            
            if it % 10 == 0:
                utils.print_log('main Train, {}, {}, {:.6f}, {:.3f}, {:.3f}'
                      .format(epoch, it, loss.item(), 100.*correct_Noise / total, 100.*correct_Real / total))
                print('main Train, {}, {}, {:.6f}, {:.3f}, {:.3f}'
                      .format(epoch, it, loss.item(), 100.*correct_Noise / total, 100.*correct_Real / total))


        # test
        state_info.sample.eval()
        total = torch.tensor(0, dtype=torch.float32)
        for it, (Noise, Ny, label_Ny) in enumerate(Noise_Test_loader):

            Noise, Ny, label_Ny = to_var(Noise, FloatTensor), to_var(Ny, LongTensor), to_var(label_Ny, LongTensor)

            Nout = state_info.forward_Sample(Noise, gamma=1)

            _, pred = torch.max(Nout.data, 1)
            correct_Test += float(pred.eq(label_Ny.data).cpu().sum())
            total += float(Noise.size(0))

        utils.print_log('main Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correct_Test / total))
        print('main Test, {}, {}, {:.3f}'
              .format(epoch, it, 100.*correct_Test / total))

        if 100.*correct_Test / total > best_prec_result:
            best_prec_result = 100.*correct_Test / total
            filename = 'checkpoint_best.pth.tar'
            utils.save_state_checkpoint(state_info, best_prec_result, epoch, mode, filename, utils.default_model_dir)

        filename = 'latest.pth.tar'
        utils.save_state_checkpoint(state_info, best_prec_result, epoch, mode, filename, utils.default_model_dir)
        state_info.lr_Sample.step()
        utils.print_log('')

    now = time.gmtime(time.time() - start_time)
    utils.print_log('Best Prec : {:.4f}'.format(best_prec_result.item()))
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))



# adversarial_loss = torch.nn.BCELoss()
# criterion_GAN = torch.nn.MSELoss()
# criterion_cycle = torch.nn.L1Loss()
# criterion_identity = torch.nn.L1Loss()
# criterion = nn.CrossEntropyLoss().cuda()