import argparse
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import os
import torch.backends.cudnn as cudnn
import time
import utils
import dataset
import math

def to_var(x, dtype):
    return Variable(x.type(dtype))

def train_disc(state_info, True_loader, Fake_loader, epoch): # all 
    
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    criterion_GAN = torch.nn.BCELoss()

    # criterion_GAN = torch.nn.MSELoss()

    utils.print_log('Type, Epoch, Batch, loss, BCE, KLD, CE')
    state_info.set_train_mode()
    correct = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)

    valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
    unvalid = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

    for epoch in range(start_epoch, epoch):

        # train
        for it, ((real, Ry, label_Ry), (fake, Fy, label_Fy)) in enumerate(zip(True_loader, Fake_loader)):

            real, Ry, label_Ry = to_var(real, FloatTensor), to_var(Ry, LongTensor), to_var(label_Ry, LongTensor)
            fake, Fy, label_Fy = to_var(fake, FloatTensor), to_var(Fy, LongTensor), to_var(label_Fy, LongTensor)

            Rout, Fout = state_info.forward_disc(real, Ry, fake, Fy)

            state_info.optim_Disc.zero_grad()
            loss_real = criterion_GAN(Rout, valid)
            loss_fake = criterion_GAN(Fout, unvalid)
            loss_Disc = (loss_real + loss_fake) / 2
            loss_Disc.backward()
            state_info.optim_Disc.step()

            '''
            Rout, Fout 결과 : value(0~1) 맞다, 틀리다 결정
            label_Ry, label_Fy : True label
            Ry, Fy : Ry는 True label, Fy는 90퍼센트 틀린 label

            맞춘 개수를 확인하고자 하면
            
            resultR = label_Ry.eq(Ry).cpu().type(torch.ByteTensor) shape : [Batch, 1]
            predR = torch.round(Rout).cpu().type(torch.ByteTensor) shape : [Batch, 1]

            resultF = label_Fy.eq(Fy).cpu().type(torch.ByteTensor) shape : [Batch, 1]
            predF = torch.round(Fout).cpu().type(torch.ByteTensor) shape : [Batch, 1]
            
            correctR += float(predR.eq(resultR.data).cpu().sum())
            correctF += float(predR.eq(resultR.data).cpu().sum())




            '''

            # 어떤 체크를 해야하는가?
            # total을 무엇으로 측정하는가? 
            # Training 하는 데이터 예측 결과를 표현하면 되겠네
            # Fy는 90퍼센트 정도 예상되니까 label_Fy랑 Fy 모두 결과를 내보자
            # Ry도 마찬가지로 결과를 측정해보자
            # test 함수에는 어떤 내용을 넣어야 하는가
            # test <- 45000 장의 이미지를 불러온다. 
            # 정확도를 측정한다.
            # 

            

            _, clsS = torch.max(clsS.data, 1)
            clsS = to_var(clsS, LongTensor)

            #  Log Print
            total += float(clsT.size(0))
            _, predicted = torch.max(clsT.data, 1)

            correct += float(predicted.eq(clsS.data).cpu().sum())

            if it % 10 == 0:
                utils.print_log('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'
                      .format(epoch, it, loss.item(), BCE, KLD, CE, 100.*correct / total))
                print('Train, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'
                      .format(epoch, it, loss.item(), BCE, KLD, CE, 100.*correct / total))

        # test
        for it, (Noise, Ny, label_Ny) in enumerate(Noise_Test_loader):



    utils.print_log('')


def train_disc(state_info, Source_train_loader, Target_train_loader, criterion_GAN, criterion_cycle, criterion_identity, criterion, epoch): # all 

    utils.print_log('Type, Epoch, Batch, G-GAN, G-CYCLE, G-ID, G-CLASS, D-A, D-B, accREAL, ~loss, accRECOV, ~loss, accTAR, ~loss')

    state_info.set_train_mode()
    correct_real = torch.tensor(0, dtype=torch.float32)
    correct_recov = torch.tensor(0, dtype=torch.float32)
    correct_target = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)

    for it, ((real_A, y), (real_B, _)) in enumerate(zip(Source_train_loader, Target_train_loader)):
        
        batch_size = real_A.size(0)
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        


        real_A, y = to_var(real_A, FloatTensor), to_var(y, LongTensor)
        real_B = to_var(real_B, FloatTensor)

        # -----------------------
        #  Train Source Classifier
        # -----------------------

        state_info.optimizer_CS.zero_grad()
        output_cls_real = state_info.cls_src(real_A) # Classifier
        loss_cls_clear = criterion(output_cls_real, y)
        loss_cls_clear.backward()
        state_info.optimizer_CS.step()

        # -----------------------
        #  Train Generator AB and BA
        # -----------------------

        state_info.optimizer_G_AB.zero_grad()
        state_info.optimizer_G_BA.zero_grad()

        # Identity loss
        loss_idt_A = criterion_identity(state_info.G_BA(real_A), real_A)
        loss_idt_B = criterion_identity(state_info.G_AB(real_B), real_B)

        loss_identity = args.identity * (loss_idt_A + loss_idt_B) / 2

        # GAN loss
        fake_B = state_info.G_AB(real_A)
        loss_GAN_AB = criterion_GAN(state_info.D_B(fake_B), valid)
        fake_A = state_info.G_BA(real_B)
        loss_GAN_BA = criterion_GAN(state_info.D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = state_info.G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = state_info.G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = args.cycle * (loss_cycle_A + loss_cycle_B) / 2

        # Class Consistency
        output_cls_recov = state_info.cls_src(recov_A) # Classifier
        loss_cls_recov = args.cls * criterion(output_cls_recov, y)

        # Total loss
        loss_G = loss_GAN + loss_cycle + loss_identity + loss_cls_recov

        loss_G.backward(retain_graph=True)
        state_info.optimizer_G_AB.step()
        state_info.optimizer_G_BA.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        state_info.optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(state_info.D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.query(fake_A)
        loss_fake = criterion_GAN(state_info.D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        state_info.optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        state_info.optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(state_info.D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.query(fake_B)
        loss_fake = criterion_GAN(state_info.D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        state_info.optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # -----------------------
        #  Train Target Classifier
        # -----------------------

        state_info.optimizer_CT.zero_grad()
        output_cls_target = state_info.cls_target(fake_B) # Classifier
        loss_cls_fake = criterion(output_cls_target, y)
        loss_cls_fake.backward()
        state_info.optimizer_CT.step()

        # -----------------------
        #  Log Print
        # -----------------------

        # output_cls_real, output_cls_recov, output_cls_target, 

        total += float(batch_size)
        _, predicted_real = torch.max(output_cls_real.data, 1)
        correct_real += float(predicted_real.eq(y.data).cpu().sum())

        _, predicted_recov = torch.max(output_cls_recov.data, 1)
        correct_recov += float(predicted_recov.eq(y.data).cpu().sum())

        _, predicted_target = torch.max(output_cls_target.data, 1)
        correct_target += float(predicted_target.eq(y.data).cpu().sum())

        if it % 10 == 0:
            utils.print_log('Train, {}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}, {:.4f}, {:.2f}, {:.4f}, {:.2f}, {:.4f}'
                  .format(epoch, it, loss_GAN.item(), loss_cycle.item(), loss_identity.item(), loss_cls_recov.item(), loss_D_A.item(), loss_D_B.item()
                    , 100.*correct_real / total, loss_cls_clear.item(), 100.*correct_recov / total, loss_cls_recov.item(), 100.*correct_target / total, loss_cls_fake.item()))

            print('Train, {}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}, {:.4f}, {:.2f}, {:.4f}, {:.2f}, {:.4f}'
                  .format(epoch, it, loss_GAN.item(), loss_cycle.item(), loss_identity.item(), loss_cls_recov.item(), loss_D_A.item(), loss_D_B.item()
                    , 100.*correct_real / total, loss_cls_clear.item(), 100.*correct_recov / total, loss_cls_recov.item(), 100.*correct_target / total, loss_cls_fake.item()))

    utils.print_log('')

def test(state_info, Source_test_loader, Target_test_loader, criterion, realA_sample, realB_sample, epoch):
    
    utils.print_log('Type, Epoch, Batch, accSource, accTarget')
    state_info.set_test_mode()
    correct_src = torch.tensor(0, dtype=torch.float32)
    correct_target = torch.tensor(0, dtype=torch.float32)
    total = torch.tensor(0, dtype=torch.float32)
    total_loss_src = 0
    total_loss_target = 0

    if realA_sample.size(1) == 1:
        realA_sample = torch.cat([realA_sample, realA_sample, realA_sample], 1)

    if realB_sample.size(1) == 1:
        realB_sample = torch.cat([realB_sample, realB_sample, realB_sample], 1)    

    for it, ((real_A, Source_y), (real_B, Target_y)) in enumerate(zip(Source_test_loader, Target_test_loader)):

        if real_B.size(0) != real_A.size(0):
            continue

        if real_A.size(1) == 1:
            real_A = torch.cat([real_A, real_A, real_A], 1)

        if real_B.size(1) == 1:
            real_B = torch.cat([real_B, real_B, real_B], 1)
        
        batch_size = real_A.size(0)

        real_A, Source_y = to_var(real_A, FloatTensor), to_var(Source_y, LongTensor)
        real_B, Target_y = to_var(real_B, FloatTensor), to_var(Target_y, LongTensor)

        
        output_cls_src = state_info.cls_src(real_A) # Classifier
        # loss_cls_src = criterion(output_cls_src, Source_y)        

        output_cls_target = state_info.cls_target(real_B) # Classifier
        # loss_cls_target = criterion(output_cls_target, Target_y)

        total += float(batch_size)
        _, predicted_src = torch.max(output_cls_src.data, 1)
        correct_src += float(predicted_src.eq(Source_y.data).cpu().sum())

        _, predicted_target = torch.max(output_cls_target.data, 1)
        correct_target += float(predicted_target.eq(Target_y.data).cpu().sum())

    make_sample_image(state_info, epoch, realA_sample, realB_sample) # img_gen_src, Source_y, img_gen_target, Target_y

    source_prediction_max_result.append(correct_src)
    target_prediction_max_result.append(correct_target)

    utils.print_log('Test, {}, {}, {:.2f}, {:.2f}'.format(epoch, it, 100.*correct_src / total, 100.*correct_target / total))
    print('Test, {}, {}, {:.2f}, {:.2f}'.format(epoch, it, 100.*correct_src / total, 100.*correct_target / total))

    return 100.*correct_target / total
