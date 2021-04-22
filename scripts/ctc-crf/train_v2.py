"""
Copyright 2020 Tsinghua University
Apache 2.0.
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)

This script uses DistributedDataParallel (DDP) to train model within framework of CAT.
Differed from `train_dist.py`, this one supports read configurations from json file
and is more non-hard-coding style.
"""

from collections import OrderedDict
from ctc_crf import CTC_CRF_LOSS as CRFLoss
from ctc_crf import WARP_CTC_LOSS as CTCLoss
from monitor import plot_monitor
import utils
import model as model_zoo
import dataset as DataSet
import os
import argparse
import json
import shutil
import time
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

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
        sampler=train_sampler, collate_fn=DataSet.sortedPadCollate)

    testloader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=test_sampler, collate_fn=DataSet.sortedPadCollate)

    logger = OrderedDict({
        'log_train': ['epoch,loss,loss_real,net_lr,time'],
        'log_eval': ['loss_real,time']
    })
    manager = Manager(logger, args)

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
    if args.rank == 0:
        plot_monitor(args.dir.split('/')[-1])

    if args.iscrf:
        ctc_crf_base.release_env(gpus)


class Manager(object):
    def __init__(self, logger: OrderedDict, args):
        super().__init__()

        with open(args.config, 'r') as fi:
            configures = json.load(fi)

        self.model = build_model(args, configures)

        self.scheduler = utils.GetScheduler(
            configures['scheduler'], self.model.parameters())

        self.log = logger
        self.rank = args.rank
        self.DEBUG = args.debug

        if args.resume is not None:
            print(f"Resuming from: {args.resume}")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            self.load(checkpoint)

    def run(self, train_sampler, trainloader, testloader, args):

        epoch = self.scheduler.epoch_cur
        self.model.train()
        while True:
            epoch += 1
            train_sampler.set_epoch(epoch)

            train(trainloader, epoch, args, self)

            self.model.eval()
            metrics = test(testloader, args, self)
            if isinstance(metrics, tuple):
                # defaultly use the first one to evaluate
                metrics = metrics[0]
            state = self.scheduler.step(epoch, metrics)

            self.model.train()
            if self.rank == 0:
                self.log_export(args.ckptpath)

            if state == 2:
                print("Break: GPU[%d]" % self.rank)
                dist.barrier()
                break
            elif self.rank != 0:
                continue
            elif state == 0 or state == 1:
                self.save("checkpoint", args.ckptpath)
                if state == 1 and not self.DEBUG:
                    shutil.copyfile(
                        f"{args.ckptpath}/checkpoint.pt", f"{args.ckptpath}/bestckpt.pt")

                    # save model for inference
                    torch.save(self.model.module.infer.state_dict(),
                               args.ckptpath + "/infer.pt")
            else:
                raise ValueError(f"Unknown state: {state}.")
            torch.cuda.empty_cache()

    def save(self, name, PATH=''):
        """Save checkpoint.

        The checkpoint file would be located at `PATH/name.pt`
        or `name.pt` if `PATH` is empty.
        """

        if self.DEBUG:
            utils.highlight_msg("Debugging, skipped saving model.")
            return None

        if PATH != '' and PATH[-1] != '/':
            PATH += '/'
        torch.save(OrderedDict({
            'model': self.model.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'log': OrderedDict(self.log)
        }), PATH+name+'.pt')

    def load(self, checkpoint):
        r'Load checkpoint.'

        dist.barrier()
        self.model.load_state_dict(checkpoint['model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.log = checkpoint['log']

    def log_update(self, msg=[], loc="log_train"):
        assert loc in self.log

        self.log[loc].append(msg)

    def log_export(self, PATH):
        """Save log file in {PATH}/{key}.csv
        """
        if self.DEBUG:
            utils.highlight_msg("Debugging, skipped log dump.")
            return None

        for key, value in self.log.items():

            with open(f"{PATH}/{key}.csv", 'w+', encoding='utf8') as file:
                data = [','.join([str(x) for x in infos])
                        for infos in value[1:]]
                file.write(value[0] + '\n' + '\n'.join(data))


def train(trainloader, epoch: int, args, manager: Manager):

    scheduler = manager.scheduler

    model = manager.model
    optimizer = scheduler.optimizer

    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    losses_real = utils.AverageMeter('Loss_real', ':.4e')
    progress = utils.ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses, losses_real],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()

    for i, minibatch in enumerate(trainloader):
        if args.debug and i > 20:
            utils.highlight_msg("In debug mode, quit training.")
            dist.barrier()
            break
        # measure data loading time
        logits, input_lengths, labels, label_lengths, path_weights = minibatch

        data_time.update(time.time() - end)

        optimizer.zero_grad()

        logits, labels, input_lengths, label_lengths = logits.cuda(
            args.gpu, non_blocking=True), labels, input_lengths, label_lengths

        loss = model(logits, labels, input_lengths, label_lengths)

        with torch.no_grad():
            if args.iscrf:
                partial_loss = loss.cpu()
                weight = torch.mean(path_weights)
                real_loss = partial_loss - weight
            else:
                real_loss = loss.cpu()

        # measure accuracy and record loss
        losses.update(loss.item(), logits.size(0))
        losses_real.update(real_loss.item(), logits.size(0))

        loss.backward()

        optimizer.step()
        scheduler.update_lr((epoch - 1) * len(trainloader) + i + 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        manager.log_update(
            [epoch, loss.item(), real_loss.item(), optimizer.param_groups[0]['lr'], time.time() - end], loc='log_train')

        end = time.time()

        if i % args.print_freq == 0 or args.debug:
            progress.display(i)


def test(testloader, args, manager: Manager):

    model = manager.model

    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses_real = utils.AverageMeter('Loss_real', ':.4e')
    progress = utils.ProgressMeter(
        len(testloader),
        [batch_time, data_time, losses_real],
        prefix='Test: ')

    beg = time.time()
    end = time.time()
    with torch.no_grad():
        for i, minibatch in enumerate(testloader):
            if args.debug and i > 20:
                utils.highlight_msg("In debug mode, quit evaluating.")
                dist.barrier()
                break
            # measure data loading time
            logits, input_lengths, labels, label_lengths, path_weights = minibatch
            data_time.update(time.time() - end)

            logits, labels, input_lengths, label_lengths = logits.cuda(
                args.gpu, non_blocking=True), labels, input_lengths, label_lengths
            path_weights = path_weights.cuda(args.gpu, non_blocking=True)

            loss = model(logits, labels, input_lengths, label_lengths)

            if args.iscrf:
                weight = torch.mean(path_weights)
                real_loss = loss - weight
            else:
                real_loss = loss

            dist.all_reduce(real_loss, dist.ReduceOp.SUM)
            real_loss = real_loss / torch.cuda.device_count()

            # measure accuracy and record loss
            losses_real.update(real_loss.item(), logits.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            end = time.time()

            if i % args.print_freq == 0 or args.debug:
                progress.display(i)

    manager.log_update(
        [losses_real.avg, time.time() - beg], loc='log_eval')

    return losses_real.avg


class CAT_Model(nn.Module):
    def __init__(self, NET=None, fn_loss='crf', lamb: float = 0.1, net_kwargs: dict = None):
        super().__init__()
        if NET is None:
            return None

        self.infer = NET(**net_kwargs)

        if fn_loss == "ctc":
            self.loss_fn = CTCLoss()  # torch.nn.CTCLoss()
        elif fn_loss == "crf":
            self.loss_fn = CRFLoss(lamb=lamb)
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
        netconfigs['lossfn'] = 'crf'
        print(
            "Warning: not specified \'lossfn\' in configuration, defaultly set to \'crf\'")
    if netconfigs['lossfn'] == 'crf' and 'lamb' not in netconfigs:
        netconfigs['lamb'] = 0.01
        print("Warning: not specified \'lamb\' in configuration, defaultly set to 0.01")

    setattr(args, 'iscrf', netconfigs['lossfn'] == 'crf')
    model = CAT_Model(
        net, netconfigs['lossfn'], netconfigs['lamb'], net_kwargs)

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
