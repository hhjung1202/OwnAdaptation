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
    mode = args.model
    utils.default_model_dir = os.path.join(args.dir, mode)
    
    start_epoch = 0
    checkpoint = None
    checkpoint = utils.load_checkpoint(utils.default_model_dir)    
    if not checkpoint:
        args.last_epoch = -1
        state_info.learning_scheduler_init(args, mode)
    else:
        state_info.load_state_dict(checkpoint, mode)
        state_info.learning_scheduler_init(args, mode)

    for epoch in range(0, args.epoch):

        epoch_result = train(args, state_info, Train_loader, Test_loader, epoch)
        
        if epoch_result > best_prec_result:
            best_prec_result = epoch_result
            utils.save_state_checkpoint(state_info, best_prec_result, epoch, 'checkpoint_best.pth.tar', utils.default_model_dir)
            print('save..')

        state_info.lr_model.step()
        utils.print_log('')

    now = time.gmtime(time.time() - start_time)
    utils.print_log('Best Prec : {:.4f}'.format(best_prec_result.item()))
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))