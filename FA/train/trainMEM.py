import torch
import numpy as np
import os
import time
import utils
from .trainFunction import *

def train_MEM(args, state_info, Train_loader, Test_loader, AnchorSet): # all 

    start_time = time.time()
    best_prec_result = torch.tensor(0, dtype=torch.float32)
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    mode = args.model
    utils.default_model_dir = os.path.join(args.dir, mode)
    
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

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

        epoch_result = train(args, state_info, Train_loader, Test_loader, Memory, criterion, epoch)
        if epoch_result > best_prec_result:
            best_prec_result = epoch_result
        state_info.lr_model.step()
        utils.print_log('')

    now = time.gmtime(time.time() - start_time)
    utils.print_log('Best Prec : {:.4f}'.format(best_prec_result.item()))
    utils.print_log('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))