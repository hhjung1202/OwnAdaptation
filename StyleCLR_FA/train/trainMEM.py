import torch
import numpy as np
import os
import time
import utils
from .trainFunction import *
import torch.utils.data as data

def train_MEM(args, state_info, Train_loader, Test_loader): # all 

    start_time = time.time()
    best_prec_result = torch.tensor(0, dtype=torch.float32)
    best_prec_result2 = torch.tensor(0, dtype=torch.float32)
    mode = args.model
    utils.default_model_dir = os.path.join(args.dir, mode)
    
    start_epoch = 0
    checkpoint = None
    # checkpoint = utils.load_checkpoint(utils.default_model_dir)    
    if not checkpoint:
        args.last_epoch = -1
        state_info.learning_scheduler_init(args, mode)
    else:
        start_epoch = checkpoint['epoch'] + 1
        best_prec_result = checkpoint['Best_Prec']
        state_info.load_state_dict(checkpoint, mode)
        args.last_epoch = start_epoch
        state_info.learning_scheduler_init(args, mode)

    for epoch in range(0, args.epoch):

        train(args, state_info, Train_loader, Test_loader, epoch)
        # epoch_result, epoch_result2 = train(args, state_info, Train_loader, Test_loader, epoch)
        
        # if epoch_result > best_prec_result:
        #     best_prec_result = epoch_result

        # if epoch_result2 > best_prec_result2:
        #     best_prec_result2 = epoch_result2

        state_info.lr_model.step()
        utils.print_log('')

    now = time.gmtime(time.time() - start_time)
    # utils.print_log('Best Prec : {:.4f}'.format(best_prec_result.item()))
    # utils.print_log('Best Prec : {:.4f}'.format(best_prec_result2.item()))
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))