'''
Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
Apache 2.0.
This script shows how to excute CTC-CRF neural network training with PyTorch.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import datetime
import torch.nn.functional as F
import numpy as np
import timeit
import os
import sys
import gc
import csv
from torch.autograd import Variable
from torch.autograd import Function
from ctc_crf import CTC_CRF_LOSS
from torch.utils.data import Dataset, DataLoader
from model import CAT_RegModel, CAT_Chunk_Model
from dataset import SpeechDatasetMemPickel, PadCollateChunk
from log_utils import params_num, init_logging
from plot_train_process import plot_train_figure
from utils import save_ckpt, adjust_lr, parse_args, train_chunk_model, validate_chunk_model
import ctc_crf_base

TARGET_GPUS = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")))]
gpus = torch.IntTensor(TARGET_GPUS)

header = ['epoch', 'time', 'lr', 'held-out loss']


def train():
    args = parse_args()
   
    args_msg = ['  %s: %s' % (name, value)
                for (name, value) in vars(args).items()]
    logger.info('args:\n' + '\n'.join(args_msg))

    ckpt_path = "models_chunk_twin_context"
    os.system("mkdir -p {}".format(ckpt_path))
    logger = init_logging(
        "chunk_model", "{}/train.log".format(ckpt_path))

    csv_file = open(args.csv_file, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)

    batch_size = args.batch_size
    device = torch.device("cuda:0")

    reg_weight = args.reg_weight

    ctc_crf_base.init_env(args.den_lm_fst_path, gpus)

    model = CAT_Chunk_Model(args.feature_size, args.hdim, args.output_unit,
                            args.dropout, args.lamb, reg_weight)

    lr = args.origin_lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch = 0
    prev_cv_loss = np.inf
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        prev_cv_loss = checkpoint['cv_loss']
        model.load_state_dict(checkpoint['model'])

    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)

    reg_model = CAT_RegModel(args.feature_size, args.hdim,
                             args.output_unit, args.dropout, args.lamb)

    loaded_reg_model = torch.load(args.regmodel_checkpoint)
    reg_model.load_state_dict(loaded_reg_model)

    reg_model.cuda()
    reg_model = nn.DataParallel(reg_model)
    reg_model.to(device)

    prev_epoch_time = timeit.default_timer()

    model.train()
    reg_model.eval()
    while True:
        # training stage
        epoch += 1
        gc.collect()

        if epoch > 2:
            cate_list = list(range(1, args.cate, 1))
            random.shuffle(cate_list)
        else:
            cate_list = range(1, args.cate, 1)

        for cate in cate_list:
            pkl_path = args.tr_data_path + "/" + str(cate)+".pkl"
            if not os.path.exists(pkl_path):
                continue
            tr_dataset = SpeechDatasetMemPickel(pkl_path)

            jitter = random.randint(-args.jitter_range, args.jitter_range)
            chunk_size = args.default_chunk_size + jitter

            tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size,
                                       shuffle=True, num_workers=0, collate_fn=PadCollateChunk(chunk_size))

            train_chunk_model(model, reg_model, tr_dataloader,
                              optimizer, epoch, chunk_size, TARGET_GPUS, args, logger)

        # cv stage
        model.eval()
        cv_losses_sum = []
        cv_cls_losses_sum = []
        count = 0
        cate_list = range(1, args.cate, 1)
        for cate in cate_list:
            pkl_path = args.dev_data_path + "/"+str(cate)+".pkl"
            if not os.path.exists(pkl_path):
                continue
            cv_dataset = SpeechDatasetMemPickel(pkl_path)
            cv_dataloader = DataLoader(cv_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=0, collate_fn=PadCollateChunk(args.default_chunk_size))
            validate_count = validate_chunk_model(model, reg_model, cv_dataloader, epoch,
                                                  cv_losses_sum, cv_cls_losses_sum, args, logger)
            count += validate_count
        cv_loss = np.sum(np.asarray(cv_losses_sum))/count
        cv_cls_loss = np.sum(np.asarray(cv_cls_losses_sum))/count
        # save model
        save_ckpt({
            'cv_loss': cv_loss,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr,
            'epoch': epoch
        }, epoch < args.min_epoch or cv_loss <= prev_cv_loss, ckpt_path, "model.epoch.{}".format(epoch))

        csv_row = [epoch, (timeit.default_timer() -
                           prev_epoch_time)/60, lr, cv_loss]
        prev_epoch_time = timeit.default_timer()
        csv_writer.writerow(csv_row)
        csv_file.flush()
        plot_train_figure(args.csv_file, args.figure_file)

        if epoch < args.min_epoch or cv_loss <= prev_cv_loss:
            prev_cv_loss = cv_loss

        lr = adjust_lr(optimizer, args.origin_lr, lr, cv_loss, prev_cv_loss,
                       epoch, args.min_epoch)
        if (lr < args.stop_lr):
            print("rank {} lr is too slow, finish training".format(
                args.rank), datetime.datetime.now(), flush=True)
            break
        model.train()

    ctc_crf_base.release_env(gpus)


if __name__ == "__main__":
    train()
