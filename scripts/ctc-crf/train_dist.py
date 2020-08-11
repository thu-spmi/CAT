#!/usr/bin/python3
'''
Copyright 2018-2019 Tsinghua University, Author: Kai Hu (sunsonhu@163.com)
Apache 2.0.
This script shows how to excute distributed training based on the pytorch DistributedDataParallel API, which was released on pytorch
1.0.0 or later. 
based on https://github.com/pytorch/examples/blob/master/imagenet/main.py.
'''

import os
import sys
import gc
import datetime
import timeit
import numpy as np
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from plot_train_process import plot_train_figure
import ctc_crf_base
from dataset import SpeechDatasetPickel, PadCollate
from model import CAT_Model
from log_utils import params_num, init_logging
from utils import save_ckpt, adjust_lr_distribute, parse_args, train, validate

header = ['epoch', 'time', 'lr', 'held-out loss']


def main_worker(gpu, ngpus_per_node, args):
    csv_file = None
    csv_writer = None

    args.gpu = gpu
    args.rank = args.start_rank + gpu
    TARGET_GPUS = [args.gpu]
    logger = None
    ckpt_path = "models"
    os.system("mkdir -p {}".format(ckpt_path))

    if args.rank == 0:
        logger = init_logging(args.model, "{}/train.log".format(ckpt_path))
        args_msg = ['  %s: %s' % (name, value)
                    for (name, value) in vars(args).items()]
        logger.info('args:\n' + '\n'.join(args_msg))

        csv_file = open(args.csv_file, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

    gpus = torch.IntTensor(TARGET_GPUS)
    ctc_crf_base.init_env(args.den_lm_fst_path, gpus)
    dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)

    model = CAT_Model(args.arch, args.feature_size, args.hdim, args.output_unit,
                      args.layers, args.dropout, args.lamb, args.ctc_crf)
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

    tr_dataset = SpeechDatasetPickel(args.tr_data_path)
    tr_sampler = DistributedSampler(tr_dataset)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.gpu_batch_size, shuffle=False, num_workers=args.data_loader_workers,
                               pin_memory=True, collate_fn=PadCollate(), sampler=tr_sampler)
    cv_dataset = SpeechDatasetPickel(args.dev_data_path)
    cv_dataloader = DataLoader(cv_dataset, batch_size=args.gpu_batch_size, shuffle=False,
                               num_workers=args.data_loader_workers, pin_memory=True, collate_fn=PadCollate())

    prev_epoch_time = timeit.default_timer()

    while True:
        # training stage
        epoch += 1
        tr_sampler.set_epoch(epoch)  # important for data shuffle
        gc.collect()
        train(model, tr_dataloader, optimizer, epoch, args, logger)
        cv_loss = validate(model, cv_dataloader, epoch, args, logger)
        # save model
        if args.rank == 0:
            save_ckpt({
                'cv_loss': cv_loss,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr': lr,
                'epoch': epoch
            }, cv_loss <= prev_cv_loss, ckpt_path, "model.epoch.{}".format(epoch))

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

    ctc_crf_base.release_env(gpus)


'''
You need to start this py file on several servers, e.g. If you have three devices with 4 gpu
on each device you should start your training on all devices. The IP 10.20.101.31 must be the the IP
of the device rank0 was running on
device0: python3 train_dist.py --ctc_crf --dist-url='tcp://10.20.101.31:23457' --tr_data_path='your training data pkl file path' --dev_data_path='your validate data pkl file path' --den_lm_fst_path="your denominator fst path" --gpu_batch_size=128 --world_size=12  --start_rank=0 --output_unit=46 
device1: python3 train_dist.py --ctc_crf --dist-url='tcp://10.20.101.31:23457' --tr_data_path='your training data pkl file path' --dev_data_path='your validate data pkl file path' --den_lm_fst_path="your denominator fst path" --gpu_batch_size=128 --world_size=12  --start_rank=4 --output_unit=46 
device2: python3 train_dist.py --ctc_crf --dist-url='tcp://10.20.101.31:23457' --tr_data_path='your training data pkl file path' --dev_data_path='your validate data pkl file path' --den_lm_fst_path="your denominator fst path" --gpu_batch_size=128 --world_size=12  --start_rank=8 --output_unit=46 
'''


def main():
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    plot_train_figure(args.csv_file, args.figure_file)


if __name__ == "__main__":
    main()
