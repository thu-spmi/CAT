'''
Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang 
Apache 2.0.
This script shows how to excute CTC-CRF neural network training with PyTorch.
'''

import random
import datetime
import timeit
import shutil
import os
import math
import csv
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import ctc_crf_base
from plot_train_process import plot_train_figure
from ctc_crf import CTC_CRF_LOSS
from torch.autograd import Function
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from model import CAT_RegModel, CAT_Chunk_Model
from dataset import SpeechDatasetMemPickel, PadCollateChunk
from log_utils import params_num, init_logging
from utils import save_ckpt, adjust_lr_distribute, parse_args, train_chunk_model, validate_chunk_model

header = ['epoch', 'time', 'lr', 'held-out loss']

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.rank = args.start_rank + gpu
    TARGET_GPUS = [args.gpu]
    gpus = torch.IntTensor(TARGET_GPUS)
    logger = None
    ckpt_path = "models_chunk_twin_context"
    os.system("mkdir -p {}".format(ckpt_path))
    if args.rank == 0:
        logger = init_logging(
            "chunk_model", "{}/train.log".format("models_chunk_twin_context"))
        args_msg = ['  %s: %s' % (name, value)
                    for (name, value) in vars(args).items()]
        logger.info('args:\n' + '\n'.join(args_msg))

        csv_file = open(args.csv_file, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

    ctc_crf_base.init_env(args.den_lm_fst_path, gpus)
    #print("rank {} init process grop".format(args.rank),
    #      datetime.datetime.now(), flush=True)
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)

    model = CAT_Chunk_Model(args.feature_size, args.hdim, args.output_unit,
                            args.dropout, args.lamb, args.reg_weight, args.ctc_crf)
    if args.rank == 0:
        params_msg = params_num(model)
        logger.info('\n'.join(params_msg))

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

    model.cuda(args.gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=TARGET_GPUS)

    reg_model = CAT_RegModel(args.feature_size, args.hdim,
                             args.output_unit, args.dropout, args.lamb)
    loaded_reg_model = torch.load(args.regmodel_checkpoint)
    reg_model.load_state_dict(loaded_reg_model)
    reg_model.cuda(args.gpu)
    reg_model = nn.parallel.DistributedDataParallel(
        reg_model, device_ids=TARGET_GPUS)

    model.train()
    reg_model.eval()
    prev_epoch_time = timeit.default_timer()
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
            batch_size = int(args.gpu_batch_size * 2 / cate)
            if batch_size < 2:
                batch_size = 2
            #print("rank {} pkl path {} batch size {}".format(
            #    args.rank, pkl_path, batch_size))
            tr_dataset = SpeechDatasetMemPickel(pkl_path)
            if tr_dataset.__len__() < args.world_size:
                continue
            jitter = random.randint(-args.jitter_range, args.jitter_range)
            chunk_size = args.default_chunk_size + jitter
            tr_sampler = DistributedSampler(tr_dataset)
            tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size,
                                       shuffle=False, num_workers=0, collate_fn=PadCollateChunk(chunk_size), drop_last=True,  sampler=tr_sampler)
            tr_sampler.set_epoch(epoch)  # important for data shuffle
            print("rank {} lengths_cate: {}, chunk_size: {}, training epoch: {}".format(
                args.rank, cate, chunk_size, epoch))
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
            batch_size = int(args.gpu_batch_size * 2 / cate)
            if batch_size < 2:
                batch_size = 2
            cv_dataset = SpeechDatasetMemPickel(pkl_path)
            cv_dataloader = DataLoader(cv_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=0, collate_fn=PadCollateChunk(args.default_chunk_size), drop_last=True)
            validate_count = validate_chunk_model(model, reg_model, cv_dataloader, epoch,
                                 cv_losses_sum, cv_cls_losses_sum, args, logger)
            count += validate_count

        cv_loss = np.sum(np.asarray(cv_losses_sum))/count
        cv_cls_loss = np.sum(np.asarray(cv_cls_losses_sum))/count

        #print("mean_cv_loss:{} , mean_cv_cls_loss: {}".format(cv_loss, cv_cls_loss))
        if args.rank == 0:
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
        else:
            args.annealing_epoch = 0

        lr = adjust_lr_distribute(optimizer, args.origin_lr, lr, cv_loss, prev_cv_loss,
                                  epoch, args.annealing_epoch, args.gpu_batch_size, args.world_size)
        if (lr < args.stop_lr):
            print("rank {} lr is too slow, finish training".format(
                args.rank), datetime.datetime.now(), flush=True)
            break

        model.train()

    ctc_crf_base.release_env(gpus)


def main():
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    plot_train_figure(args.csv_file, args.figure_file)


if __name__ == "__main__":
    main()
