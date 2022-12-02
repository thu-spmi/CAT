# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""training/evaluating manager"""

from . import (
    coreutils,
    tokenizer as tknz
)
from .specaug import SpecAug
from ._constants import (
    F_CHECKPOINT_LIST,
    F_TRAINING_INFO
)
from .data import (
    DynamicBatchDistSampler,
    ReadBatchDataLoader,
    PipeTokenize
)
from .scheduler import (
    State,
    build_scheduler
)

import os
import sys
import glob
import time
import shutil
import argparse
import webdataset as wds
from braceexpand import braceexpand
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm
from typing import *

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.optim import ZeroRedundancyOptimizer


class Manager(object):
    def __init__(
            self,
            Dataset: torch.utils.data.Dataset,
            collate_fn: Callable,
            args: argparse.Namespace,
            func_build_model: Callable[[dict, argparse.Namespace], Union[nn.Module, nn.parallel.DistributedDataParallel]],
            func_train: Optional[Callable] = None,
            func_eval: Optional[Callable] = None,
            _wds_hook: Callable[[wds.WebDataset], wds.WebDataset] = None):
        """Initialize the manager for training.

        _wds_hook (callable): for webdataset loading, the dataset would be loaded as
            >>> # dataset is an instance of WebDataset
            >>> dataset = _wds_hook(dataset)
        """
        super().__init__()

        coreutils.check_parser(args, [
            'rank', 'gpu', 'workers', 'trset', 'devset', 'dynamic_batch_mode',
            'batch_size', 'grad_accum_fold', 'config', 'dir', 'debug',
            '_logdir', '_checkdir', 'resume', 'init_model'
        ])

        # setup dataloader
        val_set = Dataset(args.devset)

        setattr(args, 'n_steps', 0)
        world_size = dist.get_world_size()
        if args.large_dataset:
            assert args.tokenizer is not None, f"--tokenizer is required for --large-dataset"
            assert os.path.isfile(args.tokenizer), \
                f"--tokenizer={args.tokenizer} is not a valid file."
            # large dataset doesnot support dynamic batching
            args.dynamic_batch_mode = -1

            '''
            NOTE (Huahuan):
            1.  ref: https://github.com/tmbdev-archive/webdataset-examples/blob/master/main-wds.py
                Explicitly setting 'RANK' and 'WORLD_SIZE' is useful for webdataset to
                recognize the DDP. Without the setting, data in all nodes are the same.
                I guess this is a bug of WebDataset.
              
            2.  In DDP, commonly, all nodes should get the same size of batches, however, 
                the data size might not be divisible to the num_nodes as well as the batch size.
                so we just drop a few of data every epoch. This won't affect much since we usually 
                train lots of epochs. And if you're concerned about that, duplicating part of the dataset to 
                allow it fitting the size is OK. But that would require knowing the size of dataset and is somewhat more complicated.
            '''
            os.environ['RANK'] = str(dist.get_rank())
            os.environ['WORLD_SIZE'] = str(world_size)
            tr_set = (
                wds.WebDataset(
                    # expand expression first with braceexpand, then glob, e.g.
                    # "{a,b,c}/*.tar" -> ["a/*.tar", "b/*.tar", "c/*.tar"] -> ["a/1.tar", "a/2.tar", ...]
                    [f for p_expanded in braceexpand(args.trset)
                     for f in glob.glob(p_expanded)],
                    shardshuffle=True,
                    nodesplitter=wds.shardlists.split_by_node
                )
                # buffer size of shuffling
                .shuffle(2000)
                # decode the .tar file to normal data
                .decode()
                # extract data to original tuple
                .to_tuple("mat.npy", "label.txt")
                # convert raw text into tensor with tokenizer
                .map(PipeTokenize(tknz.load(args.tokenizer)))
            )
            if _wds_hook is not None:
                # add some hook if needed, e.g. filter short seqs for CTC/CRF
                tr_set = _wds_hook(tr_set)
            tr_set = tr_set.batched(
                args.batch_size//world_size,
                collation_fn=collate_fn,
                # set partial=False to avoid a partial batch, but would drop a few of data, see bellow disscussion.
                partial=False
            )

            trainloader = wds.WebLoader(
                tr_set, num_workers=1, shuffle=False,
                # batching is done by webdataset
                batch_size=None)
            train_sampler = None
        else:
            tr_set = Dataset(args.trset)
            if args.dynamic_batch_mode != -1 and world_size > 1:
                coreutils.distprint(
                    "> enable dynamic batching", args.gpu)
                if args.dynamic_batch_mode == 0 and args.grad_accum_fold > 1:
                    """
                    NOTE (huahuan): with dynamic batching, in batch mode, at each update, the global
                    ... batch size (g_bs) is always `args.batch_size`. However, with bucket mode,
                    ... the g_bs could be different at steps, so the
                    ... grad_accum_fold would introduce grad bias. I think this won't
                    ... affect much, because the g_bs would only vary in a small range.
                    ... That's why here is a WARNING instead of an ERROR.
                    """
                    coreutils.distprint(
                        "warning: bucket dynamic batching with --grad_accum_fold > 1 "
                        "would probably produce inconsistent results.",
                        args.gpu
                    )
                train_sampler = DynamicBatchDistSampler(
                    dataset=tr_set,
                    mode=['bucket', 'batch'][args.dynamic_batch_mode],
                    global_batch_size=args.batch_size,
                    max_bucket_size=args.dynamic_bucket_size,
                    local_rank=args.gpu
                )
                trainloader = DataLoader(
                    tr_set, batch_sampler=train_sampler,
                    num_workers=args.workers, collate_fn=collate_fn,
                    prefetch_factor=4, persistent_workers=True
                )
            else:
                args.dynamic_batch_mode = -1
                train_sampler = DistributedSampler(tr_set)
                trainloader = DataLoader(
                    tr_set, batch_size=args.batch_size//world_size,
                    num_workers=args.workers, sampler=train_sampler, collate_fn=collate_fn,
                    prefetch_factor=4, persistent_workers=True)

        if args.dynamic_batch_mode == -1:
            args.batch_size = (args.batch_size // world_size) * world_size
            trainloader = ReadBatchDataLoader(trainloader, bs=args.batch_size)
        else:
            trainloader = ReadBatchDataLoader(trainloader, dynamic=True)

        val_sampler = DistributedSampler(val_set, shuffle=False)
        valloader = DataLoader(
            val_set, batch_size=args.batch_size//world_size, shuffle=False,
            num_workers=args.workers, sampler=val_sampler,
            collate_fn=collate_fn, persistent_workers=True
        )

        self.train_sampler = train_sampler
        self.trainloader = trainloader
        self.valloader = valloader

        # Initial model
        cfg = coreutils.readjson(args.config)  # type: dict
        self.model = func_build_model(cfg, args)

        coreutils.distprint("> model built. # of params: {:.2f} M".format(
            coreutils.count_parameters(self.model)/1e6), args.gpu)

        # get GPU info and create readme.md
        # NOTE: the following function requires the allreduce OP, so don't put it inside the `if...:` block
        gpu_info = coreutils.gather_all_gpu_info(args.gpu)
        if args.rank == 0 and not args.debug:
            coreutils.gen_readme(
                os.path.join(
                    args.dir,
                    F_TRAINING_INFO
                ),
                model=self.model,
                gpu_info=gpu_info
            )

        # hook the function
        self.train = train if func_train is None else func_train
        self.evaluate = evaluate if func_eval is None else func_eval

        # Initial specaug module
        if 'specaug' not in cfg:
            specaug = None
            coreutils.distprint("> disable SpecAug", args.gpu)
        else:
            specaug = SpecAug(**cfg['specaug'])
            specaug = specaug.to(f'cuda:{args.gpu}')
        self.specaug = specaug

        # Initial scheduler and optimizer
        assert 'scheduler' in cfg
        self.scheduler = build_scheduler(
            cfg['scheduler'], self.model.parameters())

        # Initialize the grad scaler
        self.scaler = GradScaler(enabled=args.amp)

        self.rank = args.rank   # type: int
        self.DEBUG = args.debug  # type: bool
        self.epoch = 1      # type: int
        self.step = 0       # type: int
        # used to resume from checkpoint
        self.step_by_last_epoch = 0   # type: int

        if not (args.resume is None or args.init_model is None):
            coreutils.distprint(
                "warning: you specify both --resume and --init-model, "
                "but --init-model will be ignored.", args.rank)

        if args.resume is not None:
            coreutils.distprint(
                f"> resuming from: {args.resume}", args.gpu)
            checkpoint = torch.load(
                args.resume, map_location=f'cuda:{args.gpu}')  # type: OrderedDict
            self.load(checkpoint)
            del checkpoint
        elif args.init_model is not None:
            coreutils.distprint(
                f"> initialize model from: {args.init_model}", args.gpu)
            checkpoint = torch.load(
                args.init_model, map_location=f'cuda:{args.gpu}')  # type: OrderedDict

            try:
                self.model.load_state_dict(checkpoint['model'])
            except RuntimeError as re:
                if "Error(s) in loading state_dict" in str(re):
                    self.model.load_state_dict(
                        coreutils.translate_prev_checkpoint(
                            checkpoint['model'])
                    )
                else:
                    raise RuntimeError(str(re))
            del checkpoint

        # Initialize the checkpoint manager
        try:
            user = os.getlogin()
        except OSError:
            user = "defaultUser"

        self.cm = CheckManager(
            os.path.join(args._checkdir, F_CHECKPOINT_LIST),
            header=f"created by {user} at {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Initialize the tensorboard
        if args.rank == 0:
            self.writer = SummaryWriter(os.path.join(
                args._logdir, "{0:%Y%m%d-%H%M%S/}".format(datetime.now())))
        else:
            self.writer = None

    def run(self, args: argparse.Namespace):
        coreutils.check_parser(
            args, ['_checkdir', 'rank', 'gpu', 'dir'])

        self.model.train()
        terminated = False
        while not terminated:
            if self.train_sampler is None:
                pass
            else:
                self.train_sampler.set_epoch(self.epoch)
            if self.step == 0 and not self.DEBUG:
                # get the initialized perf. before training start
                self.model.eval()
                metrics = self.evaluate(self.valloader, args, self)
                self.model.train()
                coreutils.distprint(
                    f"Epoch: {self.epoch:<3} | Step: {self.step} | Eval metric: {metrics:.3e} | LR: {self.scheduler.lr_cur:.3e}",
                    args.gpu)

            for _ in self.train(self.trainloader, args, self):
                self.model.eval()
                metrics = self.evaluate(self.valloader, args, self)
                self.model.train()
                if isinstance(metrics, tuple):
                    # defaultly use the first one to evaluate
                    metrics = metrics[0]

                state = self.scheduler.step(metrics)
                checkpoint = os.path.join(
                    args._checkdir,
                    f"checkpoint.{self.epoch}e{self.step}s.pt"
                )
                # inside self.save(), there is an all_reduce OP, don't put it in rank==0 block.
                self.save(checkpoint)
                if self.rank == 0 and not self.DEBUG:
                    self.cm.appendinfo(
                        self.epoch, self.step,
                        metrics, self.scheduler.lr_cur, checkpoint)

                coreutils.distprint(
                    f"Epoch: {self.epoch:<3} | Step: {self.step} | Eval metric: {metrics:.3e} | LR: {self.scheduler.lr_cur:.3e}",
                    args.gpu)
                if state == State.TERMINATED:
                    # backup the last checkpoint
                    if self.rank == 0 and not self.DEBUG:
                        shutil.copyfile(checkpoint, os.path.join(
                            args._checkdir, "checkpoint.pt"))
                    print("Terminated: GPU[%d]" % self.rank)
                    terminated = True
                    dist.barrier()
                    break
                elif state == State.IMPROVED:
                    # maybe do something with the best model by far
                    pass
                elif state == State.CONTINUE:
                    pass
                else:
                    raise RuntimeError(f"Unknown state: {state}.")

            self.epoch += 1

    def save(self, name: str, PATH: str = '', extra_states: Optional[Union[Dict, OrderedDict]] = None) -> str:
        """Save checkpoint.

        The checkpoint file would be located at:
        `PATH/name.pt`, or `name(.pt)` if `PATH` is empty.
        """

        if isinstance(self.scheduler.optimizer, ZeroRedundancyOptimizer):
            # the ZeroRedundancyOptimizer shards the optimizer into processes,
            # so we need to collect them to save on the disk.
            self.scheduler.optimizer.consolidate_state_dict(0)

        if self.rank != 0 or self.DEBUG:
            return None

        if name[-3:] != '.pt':
            name += '.pt'
        PATH = os.path.join(PATH, name)
        states = OrderedDict({
            'model': self.model.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'scaler': self.scaler.state_dict(),
            'step': self.step,
            'step_by_last_epoch': self.step_by_last_epoch
        })
        if extra_states is not None:
            states.update(extra_states)
        torch.save(states, PATH)
        return PATH

    def load(self, checkpoint: OrderedDict, return_state: bool = False):
        r'Load checkpoint.'

        dist.barrier()
        try:
            self.model.load_state_dict(checkpoint['model'])
        except RuntimeError as re:
            if "Error(s) in loading state_dict" in str(re):
                self.model.load_state_dict(
                    coreutils.translate_prev_checkpoint(checkpoint['model'])
                )
            else:
                raise RuntimeError(str(re))

        self.scheduler.load_state_dict(checkpoint['scheduler'])

        # FIXME: This is for old ver. compatible.
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.step_by_last_epoch = checkpoint.get(
            'step_by_last_epoch', self.step)

        if return_state:
            return checkpoint
        else:
            return None


"""
NOTE (Huahuan):
    with --dynamic_batch_mode, batch size on each device might be different,
    however, torch DDP automatically makes the allreduce on gradients,
    then averages them by world size during backward.

    That assumes the batch sizes across devices are the same.
    To address this, we re-calculate the loss in a hack way:
        local_loss_new = sum_over_local_batches(local_loss) / global_batch_size * world_size

    Here we prove the new hacked-loss is equivalent to the standard one:
    Currently the loss is (in standard DDP):
        loss_normalized = sum_over_devices(mean_over_local_batches(local_loss)) / world_size
                        = sum_over_devices(sum_over_local_batches(local_loss) / (global_batch_size / world_size))  / world_size
                        = sum_over_devices(sum_over_local_batches(local_loss)) / global_batch_size


    With re-defining the local_loss, we substitute `local_loss_new` to replace `mean_over_local_batches(local_loss)`, here is
        loss_normalized_new' = sum_over_devices(sum_over_local_batches(local_loss) / global_batch_size)
                             = loss_normalized

    In this way, the gradient is properly computed. Also, be aware that this
    might cause numerical difference given the fact that probably: (f * N) / N != f
"""


def train(trainloader: ReadBatchDataLoader, args: argparse.Namespace, manager: Manager, hook_func: Callable = None):
    """
    The default train function.

    Args:
        trainloader (Dataloader)
        args (Namespace) : configurations
        manager (Manager) : the manager for pipeline control
        _trainer_hook (optional, callable function) : custom hook function, check source code for usage.
    """

    def _go_step(g_batch_size: int, minibatch) -> Tuple[torch.Tensor, int]:
        feats, frame_lens, labels, label_lens = minibatch
        feats = feats.cuda(args.gpu, non_blocking=True)
        if manager.specaug is not None:
            feats, frame_lens = manager.specaug(feats, frame_lens)

        with autocast(enabled=use_amp):
            if hook_func is None:
                loss = model(feats, labels, frame_lens, label_lens)
            else:
                # you could custom model forward, tracks logging and metric calculation in the hook
                loss = hook_func(
                    manager, model, args, i+1,
                    (feats, labels, frame_lens, label_lens)
                )
            if isinstance(loss, tuple):
                loss = loss[0]

            raw_loss = loss.detach()
            # divide loss with fold since we want the gradients to be divided by fold
            loss /= fold

        loss.data = loss.detach() * (feats.size(0) * world_size / g_batch_size)
        scaler.scale(loss).backward()

        # return for logging
        return raw_loss, feats.size(0)

    coreutils.check_parser(args, ['grad_accum_fold', 'n_steps', 'verbose',
                                  'print_freq', 'check_freq', 'rank', 'gpu', 'debug', 'amp', 'grad_norm'])

    model = manager.model
    scaler = manager.scaler
    scheduler = manager.scheduler
    optimizer = scheduler.optimizer
    optimizer.zero_grad()
    use_amp = args.amp
    grad_norm = args.grad_norm

    world_size = dist.get_world_size()
    fold = args.grad_accum_fold
    assert fold >= 1
    accum_loss = 0.
    n_batch = 0
    t_data = 0.
    t_last_step = time.time()
    t_last_batch = time.time()
    cnt_step_update = 0
    is_wds_dl = args.large_dataset
    is_quit = None
    if is_wds_dl:
        is_quit = torch.tensor(0, dtype=torch.bool, device=args.gpu)

    def get_progress_bar():
        return tqdm(
            desc=f'Epoch: {manager.epoch} | train',
            unit='batch',
            total=(args.n_steps if args.check_freq == -1 else args.check_freq),
            disable=(args.gpu != 0 or args.verbose),
            leave=False
        )
    # when check_freq > epoch size, the progress bar would display in mistake.
    p_bar = get_progress_bar()
    for i, (bs, minibatch) in enumerate(trainloader):
        # since the gradient fold could be > 1, we need to accumulate the time
        if args.verbose:
            t_data += time.time() - t_last_batch

        # skip steps when resuming from stop training
        if (cnt_step_update + manager.step_by_last_epoch < manager.step):
            if fold == 1 or (i+1) % fold == 0:
                cnt_step_update += 1
                p_bar.update()
                if args.verbose and args.gpu == 0:
                    sys.stderr.write(
                        f"\rIn skipping steps: {cnt_step_update + manager.step_by_last_epoch}/{manager.step}")
                    sys.stderr.flush()
            continue

        if is_wds_dl:
            dist.all_reduce(is_quit, op=dist.ReduceOp.MAX)
            if is_quit:
                break

        # update every fold times and drop the last few batches (number of which <= fold)
        if fold == 1 or (i+1) % fold == 0:
            local_loss, local_bs = _go_step(bs, minibatch)
            accum_loss += local_loss * local_bs
            n_batch += local_bs

            if grad_norm > 0.0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_norm, error_if_nonfinite=False)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            manager.step += 1
            scheduler.update_lr_step(manager.step)
            cnt_step_update += 1
            p_bar.update()

            # measure accuracy and record loss; item() can sync all processes.
            tolog = {
                'loss': (accum_loss/n_batch).item(),
                'lr': scheduler.lr_cur
            }

            # update tensorboard
            if args.rank == 0:
                manager.writer.add_scalar(
                    'loss/train_loss', tolog['loss'], manager.step)
                manager.writer.add_scalar(
                    'lr', tolog['lr'], manager.step)

            if args.verbose:
                coreutils.distprint(
                    f"[{manager.epoch} - {cnt_step_update}/{args.n_steps}] | data {t_data:6.3f} | time {time.time()-t_last_step:6.3f} | "
                    f"loss {tolog['loss']:.2e} | lr {tolog['lr']:.2e}",
                    args.gpu)
                t_data = 0.0
                t_last_step = time.time()

            if args.check_freq != -1 and (manager.step % args.check_freq) == 0:
                p_bar.close()
                yield None
                p_bar = get_progress_bar()

            # reset accumulated loss
            accum_loss = 0.
            n_batch = 0
        else:
            # gradient accumulation w/o sync
            with model.no_sync():
                local_loss, local_bs = _go_step(bs, minibatch)
                accum_loss += local_loss * local_bs
                n_batch += local_bs

        if args.verbose:
            t_last_batch = time.time()

    if is_wds_dl and (not is_quit):
        # set quit flag to True
        is_quit = ~is_quit
        # wait until other processes quit
        dist.all_reduce(is_quit, op=dist.ReduceOp.MAX)

    manager.step_by_last_epoch += cnt_step_update
    # update n_steps, since we don't know how many steps there are with large dataset mode.
    args.n_steps = cnt_step_update
    p_bar.close()
    if args.check_freq == -1:
        yield
    return


@torch.no_grad()
def evaluate(testloader: DataLoader, args: argparse.Namespace, manager: Manager) -> float:

    model = manager.model
    cnt_seq = 0
    total_loss = 0.

    for i, minibatch in tqdm(enumerate(testloader), desc=f'Epoch: {manager.epoch} | eval',
                             unit='batch', total=len(testloader), disable=(args.gpu != 0), leave=False):

        feats, ilens, labels, olens = minibatch
        feats = feats.cuda(args.gpu, non_blocking=True)

        '''
        Suppose the loss is reduced by mean
        '''
        loss = model(feats, labels, ilens, olens)
        if isinstance(loss, tuple):
            loss = loss[0]

        cnt_seq += feats.size(0)
        total_loss += loss * feats.size(0)

    cnt_seq = total_loss.new_tensor(cnt_seq)

    # sync info for loggin and further state control
    # NOTE: this sync is required.
    dist.all_reduce(total_loss, dist.ReduceOp.SUM)
    dist.all_reduce(cnt_seq, dist.ReduceOp.SUM)
    avg_loss = (total_loss/cnt_seq).item()

    if args.rank == 0:
        manager.writer.add_scalar('loss/dev', avg_loss, manager.step)
    return avg_loss


class CheckManager:
    def __init__(self, f_checklist: str, header: str = None) -> None:

        # the checkpoint locations would be used for identification
        '''Example
        {
            '/path/to/check000.pt': {
                'epoch': 0,
                'step':  100000,
                'metric' : 12.3,
                'lr'  :  1e-5,
                'extra' : [...]
            },
            ...
        }
        '''
        self._checks = OrderedDict()  # type: OrderedDict[str, Dict]
        self._f_checklist = f_checklist

        if header is None:
            header = ''

        if os.path.exists(f_checklist):
            # ignore the new header in case overwritten or duplicated.
            self.getcontent()
        else:
            header = '\n'.join('# '+x for x in [
                "Use '#' in a new line to identify a comment",
                "Field definition:",
                "    No.epoch No.step metric(loss) LR pathtocheckpoint ...(any append info is ok);",
                "    the float numbers are saved via (1.0).hex(), use float.fromhex('...') to get original data;",
                "    the No.step is also saved as hex via hex(123), use int('...', 16) to get original data.",
                "Header info:",
                " "*4 + header.replace('\n', ' ')
            ])
            with open(f_checklist, 'w') as fit:
                fit.write(header)

    @property
    def content(self):
        return self._checks

    def getcontent(self):
        assert os.path.isfile(self._f_checklist)

        with open(self._f_checklist, 'r') as fit:
            for line in fit:
                line = line.strip()
                if line[0] == '#' or line == '':
                    # skip the comments
                    continue
                contents = line.split()
                assert len(contents) >= 5
                n_epoch, n_step, metric, lr, f_check = contents[:5]
                self._checks.update({
                    f_check: {
                        'epoch': int(n_epoch),
                        'step': int(n_step, 16),
                        'metric': float.fromhex(metric),
                        'lr': float.fromhex(lr),
                        'extra': contents[5:]
                    }
                })

    def appendinfo(self, n_epoch: int, n_step: int, metric: float, lr: float, f_check: str, *args):
        self._checks.update({
            f_check: {
                'epoch': n_epoch,
                'step': n_step,
                'metric': metric,
                'lr': lr,
                'extra': list(args)
            }
        })
        orin_text = open(self._f_checklist, 'r').read()
        try:
            with open(self._f_checklist, 'a') as fot:
                fot.write('\n')
                fot.write((" "*4).join(
                    [
                        f"{n_epoch:04}",
                        f"{n_step:#010x}",
                        f"{float(metric).hex()}",
                        f"{float(lr).hex()}",
                        f_check
                    ]+[str(x) for x in args])
                )
        except Exception as err:
            with open(self._f_checklist, 'w') as fot:
                fot.write(orin_text)
            raise RuntimeError(str(err))
