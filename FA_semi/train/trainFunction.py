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

def train(args, state_info, labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, criterion, epoch):
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

    for batch_idx in range(args.val_iteration):
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

        inputs_x, inputs_u, targets_x = to_var(inputs_x, FloatTensor), to_var(inputs_u, FloatTensor), to_var(targets_x, LongTensor)
        label_one = FloatTensor(targets_x.size(0), 10).zero_().scatter_(1, targets_x.view(-1, 1), 1)

        state_info.optim_model.zero_grad()

        out = state_info.forward(inputs_x, inputs_u)
        # out = state_info.forward(inputs_u, inputs_x)

        loss = criterion(out, targets_x)
        loss.backward()

        state_info.optim_model.step()

        _, pred = torch.max(out.softmax(dim=1), 1)
        correct_Real += float(pred.eq(targets_x.data).cpu().sum())

        train_Size += float(x.size(0))

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.3f}'.format(epoch, it, loss.item(), 100.*correct_Real / train_Size))
            print('Train, {}, {}, {:.6f}, {:.3f}'.format(epoch, it, loss.item(), 100.*correct_Real / train_Size))

    epoch_result = test(args, state_info, test_loader, epoch)
    return epoch_result

    utils.print_log('Type, Epoch, Batch, total, Noise_Cls, Pseudo_Cls, Reg_Noise, Reg_Pseudo, Model_Real%, Pseu_Real%')
    for it, (x, y, label) in enumerate(Train_loader):

        perm = torch.randperm(x.size(0)) if args.fixed_perm else None
        x, y, label = to_var(x, FloatTensor), to_var(y, LongTensor), to_var(label, LongTensor)
        label_one = FloatTensor(y.size(0), 10).zero_().scatter_(1, y.view(-1, 1), 1)
        suffle_label, suffle_label_one = y[perm][0], label_one[perm][0]

        l = np.random.beta(0.75, 0.75)
        l = max(l, 1-l)

        mixed_label =  l * label_one + (1-l) * suffle_label_one

        state_info.optim_model.zero_grad()

        out_IN = state_info.forward_IN(x, perm)
        out_BN = state_info.forward_BN(x)

        loss_IN = { 0: criterion(out_IN, y),
                    1: criterion(out_IN, suffle_label),
                    2: soft_label_cross_entropy(out_IN, mixed_label)}[args.case]
        
        loss_BN = criterion(out_BN, y)
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
        
        perm = torch.randperm(x.size(0)) if args.fixed_perm else None

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






















    def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1,1), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        # with torch.no_grad():
        #     # compute guessed labels of unlabel samples
        #     outputs_u = model(inputs_u, is_adain=True)
        #     outputs_u2 = model(inputs_u, is_adain=True)
        #     p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
        #     pt = p**(1/args.T)
        #     targets_u = pt / pt.sum(dim=1, keepdim=True)
        #     targets_u = targets_u.detach()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        # all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        # all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
        all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # Here !!!!
        logits2 = list(torch.split(model(mixed_input, is_adain=True), batch_size)) 
        logits_x2 = logits2[0]
        logits_u2 = torch.cat(logits2[1:], dim=0)
        Lx2, Lu2, w2 = criterion(logits_x2, mixed_target[:batch_size], logits_u2, mixed_target[batch_size:], epoch+batch_idx/args.val_iteration)
        l2 = Lx2 + w2 * Lu2



        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)
        
        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/args.val_iteration)
        l1 = Lx + w * Lu

        loss = args.weight[0] * l1 + args.weight[1] * l2

        log_lx = (Lx + Lx2) /2
        log_lu = (Lu + Lu2) /2
        # --------------------------------------------

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(log_lx.item(), inputs_x.size(0))
        losses_u.update(log_lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    w=ws.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg,)

def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('{}'.format(mode), max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)