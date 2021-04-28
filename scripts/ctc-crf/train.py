"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

This script uses DistributedDataParallel (DDP) to train model within framework of CAT.
Differed from `train_dist.py`, this one supports read configurations from json file
and is more non-hard-coding style.
"""

import utils
import os
import argparse
import numpy as np
import model as model_zoo
import dataset as DataSet
from _specaug import SpecAug
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import ctc_crf_base

# This line in rid of some conditional errors.
# torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    if not torch.cuda.is_available():
        utils.highlight_msg("CPU only training is unsupported.")
        return None

    os.makedirs(args.dir+'/ckpt', exist_ok=True)
    setattr(args, 'ckptpath', args.dir+'/ckpt')

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    print(f"Global number of GPUs: {args.world_size}")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    args.rank = args.rank * ngpus_per_node + gpu
    print(f"Use GPU: local[{args.gpu}] | global[{args.rank}]")

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    args.batch_size = args.batch_size // ngpus_per_node

    print("> Data prepare")
    if args.h5py:
        data_format = "hdf5"
        utils.highlight_msg("H5py reading might cause error with Multi-GPUs.")
        Dataset = DataSet.SpeechDataset
    else:
        data_format = "pickle"
        Dataset = DataSet.SpeechDatasetPickle

    tr_set = Dataset(
        f"{args.data}/{data_format}/tr.{data_format}")
    test_set = Dataset(
        f"{args.data}/{data_format}/cv.{data_format}")
    print("Data prepared.")

    train_sampler = DistributedSampler(tr_set)
    test_sampler = DistributedSampler(test_set)
    test_sampler.set_epoch(1)

    trainloader = DataLoader(
        tr_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler, collate_fn=DataSet.sortedPadCollate())

    testloader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=test_sampler, collate_fn=DataSet.sortedPadCollate())

    logger = OrderedDict({
        'log_train': ['epoch,loss,loss_real,net_lr,time'],
        'log_eval': ['loss_real,time']
    })
    manager = utils.Manager(logger, build_model, args)

    # get GPU info
    gpu_info = utils.gather_all_gpu_info(args.gpu)

    if args.rank == 0:
        print("> Model built.")
        print("Model size:{:.2f}M".format(
            utils.count_parameters(manager.model)/1e6))

        utils.gen_readme(args.dir+'/readme.md',
                         model=manager.model, gpu_info=gpu_info)

    # init ctc-crf, args.iscrf is set in build_model
    if args.iscrf:
        gpus = torch.IntTensor([args.gpu])
        ctc_crf_base.init_env(f"{args.data}/den_meta/den_lm.fst", gpus)

    # training
    manager.run(train_sampler, trainloader, testloader, args)

    if args.iscrf:
        ctc_crf_base.release_env(gpus)


class CAT_Model(nn.Module):
    def __init__(self, NET=None, fn_loss='crf', lamb: float = 0.1, net_kwargs: dict = None, sepcaug: nn.Module = None):
        super().__init__()
        if NET is None:
            return None

        self.infer = NET(**net_kwargs)
        self.specaug = sepcaug

        if fn_loss == "ctc":
            self.loss_fn = utils.CTCLoss()
        elif fn_loss == "crf":
            self.loss_fn = utils.CRFLoss(lamb=lamb)
        else:
            raise ValueError(f"Unknown loss function: {fn_loss}")

    def forward(self, logits, labels, input_lengths, label_lengths):
        labels = labels.cpu()
        input_lengths = input_lengths.cpu()
        label_lengths = label_lengths.cpu()

        netout, lens_o = self.infer(logits, input_lengths)
        netout = torch.log_softmax(netout, dim=-1)

        loss = self.loss_fn(netout, labels, lens_o.to(
            torch.int32).cpu(), label_lengths)

        return loss


def build_model(args, configuration, train=True) -> nn.Module:

    netconfigs = configuration['net']
    net_kwargs = netconfigs['kwargs']
    net = getattr(model_zoo, netconfigs['type'])

    if not train:
        infer_model = net(**net_kwargs)
        return infer_model


    if 'lossfn' not in netconfigs:
        lossfn = 'crf'
        utils.highlight_msg(
            "Warning: not specified \'lossfn\' in configuration.\nDefaultly set to \'crf\'")
    else:
        lossfn = netconfigs['lossfn']

    if 'lamb' not in netconfigs:
        lamb = 0.01
        if lossfn == 'crf':
            utils.highlight_msg(
                "Warning: not specified \'lamb\' in configuration.\nDefaultly set to 0.01")
    else:
        lamb = netconfigs['lamb']

    if 'specaug' not in netconfigs:
        specaug = None
        if args.rank == 0:
            utils.highlight_msg("Disable SpecAug.")
    else:
        specaug = SpecAug(**netconfigs['specaug'])

    setattr(args, 'iscrf', lossfn == 'crf')
    model = CAT_Model(net, lossfn, lamb, net_kwargs, specaug)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="recognition argument")

    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Distributed Data Parallel')

    parser.add_argument("--seed", type=int, default=0,
                        help="Manual seed.")

    parser.add_argument("--resume", type=str, default=None,
                        help="Path to location of checkpoint.")

    parser.add_argument("--debug", action="store_true",
                        help="Configure to debug settings, would overwrite most of the options.")
    parser.add_argument("--h5py", action="store_true",
                        help="Load data with H5py, defaultly use pickle (recommended).")

    parser.add_argument("--config", type=str, default=None, metavar='PATH',
                        help="Path to configuration file of training procedure.")

    parser.add_argument("--data", type=str, default=None,
                        help="Location of training/testing data.")
    parser.add_argument("--dir", type=str, default=None, metavar='PATH',
                        help="Directory to save the log and model files.")

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:13943', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    if args.debug:
        utils.highlight_msg("Debugging.")

    main(args)
