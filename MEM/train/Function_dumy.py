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

    epoch_result = test(args, state_info, Test_loader, epoch)
    return epoch_result


def train_step2(args, state_info, Train_loader, Test_loader, Memory, criterion, epoch, AnchorSet):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    train_Size = torch.tensor(0, dtype=torch.float32)
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
        out, z = state_info.forward(args, x)
        _, model_pred = torch.max(out.data, 1)
        Memory.Batch_Insert(z, model_pred)

        # ------------------------------------------------------------
        _, Anchor_z = state_info.forward(args, Anchor_Image)
        Memory.Anchor_Insert(Anchor_z, Anchor_label)

        pseudo_hard_label, pseudo_soft_label, pseudo_hard_reverse_label = Memory.Calc_Pseudolabel(z)

        reg_P = Memory.get_Regularizer(z, pseudo_hard_label, reduction='mean')
        loss_N = hard_label_cross_entropy(out, y)
        loss_P_soft = soft_label_cross_entropy(out, pseudo_soft_label)
        loss_Ent = Maximize_Pseudo_Entropy_loss(pseudo_soft_label)

        # loss_Reverse_P_hard = Reverse_hard_label_cross_entropy(out, pseudo_hard_reverse_label)
        # loss_Reverse_P_soft = Reverse_soft_label_cross_entropy(out, pseudo_soft_label)

        total = loss_N + args.weight[0] * loss_P_soft + reg_P + loss_Ent

        total.backward()
        state_info.optim_model.step()

        Pseudo_Real += float(pseudo_hard_label.eq(label).sum())
        correct_Real += float(model_pred.eq(label.data).cpu().sum())
        train_Size += float(x.size(0))

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'
                  .format(epoch, it, total.item(), loss_N.item(), loss_P_soft.item(), reg_P.item(), loss_Ent.item()
                    , 100.*correct_Real / train_Size
                    , 100.*Pseudo_Real / train_Size))
            print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'
                  .format(epoch, it, total.item(), loss_N.item(), loss_P_soft.item(), reg_P.item(), loss_Ent.item()
                    , 100.*correct_Real / train_Size
                    , 100.*Pseudo_Real / train_Size))

    epoch_result = test(args, state_info, Test_loader, epoch)
    return epoch_result

def train_step3(args, state_info, Train_loader, Test_loader, Memory, criterion, epoch, AnchorSet):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    train_Size = torch.tensor(0, dtype=torch.float32)
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
        out, z = state_info.forward(args, x)
        _, model_pred = torch.max(out.data, 1)
        Memory.Batch_Insert(z, model_pred)

        # ------------------------------------------------------------
        _, Anchor_z = state_info.forward(args, Anchor_Image)
        Memory.Anchor_Insert(Anchor_z, Anchor_label)

        pseudo_hard_label, pseudo_soft_label, pseudo_hard_reverse_label = Memory.Calc_Pseudolabel(z)

        # if args.grad == "T":
        one = y.eq(pseudo_hard_label).type(FloatTensor).view(-1,1)
        zero = torch.zeros(one.size()).type(FloatTensor)
        reverse_one = one.eq(zero).type(FloatTensor).view(-1,1)

        reg_P = Memory.get_Regularizer(z, pseudo_hard_label, reduction='mean')
        loss_P_hard = hard_label_cross_entropy_same(out, pseudo_hard_label, one)
        loss_P_soft = soft_label_cross_entropy_diff(out, pseudo_soft_label, reverse_one)
        loss_Ent = Maximize_Pseudo_Entropy_loss(pseudo_soft_label)

        # loss_Reverse_P_hard = Reverse_hard_label_cross_entropy(out, pseudo_hard_reverse_label)
        # loss_Reverse_P_soft = Reverse_soft_label_cross_entropy(out, pseudo_soft_label)

        total = loss_P_hard + args.weight[0] * loss_P_soft + reg_P + args.weight[1] * loss_Ent
        # total = loss_P_hard + args.weight[1] * loss_P_soft + reg_P + loss_Ent

        total.backward()
        state_info.optim_model.step()

        Pseudo_Real += float(pseudo_hard_label.eq(label).sum())
        correct_Real += float(model_pred.eq(label.data).cpu().sum())
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