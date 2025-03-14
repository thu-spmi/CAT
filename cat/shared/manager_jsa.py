# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Sardar (sar_dar@163.com)

"""training/evaluating manager"""

from . import data, coreutils, tokenizer as tknz
from .specaug import SpecAug
from ._constants import F_CHECKPOINT_LIST, F_TRAINING_INFO, UTTS_PER_FILE
from .scheduler import State, build_scheduler
from utils.pipeline.common_utils import checkExist

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
import pickle
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.optim import ZeroRedundancyOptimizer

class Manager(object):
    def __init__(
        self,
        T_dataset: torch.utils.data.Dataset,
        collate_fn: Callable,
        args: argparse.Namespace,
        func_build_model: Callable[
            [dict, argparse.Namespace],
            Union[nn.Module, nn.parallel.DistributedDataParallel],
        ],
        func_train: Optional[Callable] = None,
        func_eval: Optional[Callable] = None,
        _wds_hook: Callable[[wds.WebDataset], wds.WebDataset] = None,
    ):
        """Initialize the manager for training.

        _wds_hook (callable): for webdataset loading, the dataset would be loaded as
            >>> # dataset is an instance of WebDataset
            >>> dataset = _wds_hook(dataset)
        """
        super().__init__()

        coreutils.check_parser(
            args,
            [
                "rank",
                "gpu",
                "workers",
                "trset",
                "devset",
                "batching_mode",
                "batching_uneven",
                "batch_size",
                "grad_accum_fold",
                "config",
                "dir",
                "debug",
                "_logdir",
                "_checkdir",
                "resume",
                "init_model",
            ],
        )

        # setup dataloader
        if isinstance(args.trset, str):
            args.trset = [args.trset]

        val_set = T_dataset(args.devset)

        setattr(args, "n_steps", 0)
        world_size = dist.get_world_size()
        if args.batching_mode == "batch" and (not args.batching_uneven):
            args.batch_size = (args.batch_size // world_size) * world_size

        if len(args.trset) == 1:
            tr_set = T_dataset(args.trset[0])
        else:
            tr_set = data.WeightedConcatDataset(
                [T_dataset(x) for x in args.trset], args.tr_set_weight
            )

        tr_sampler = data.BatchDistSampler(
            dataset=tr_set,
            mode=args.batching_mode,
            dispatch_even=(not args.batching_uneven),
            global_batch_size=args.batch_size,
            max_bucket_size=args.bucket_size,
            local_rank=args.gpu,
        )
        trainloader = DataLoader(
            tr_set,
            batch_sampler=tr_sampler,
            num_workers=args.workers,
            collate_fn=collate_fn,
        )
        trainloader = data.ReadBatchDataLoader(trainloader)
            
        val_sampler = data.BatchDistSampler(
            dataset=val_set,
            mode=args.batching_mode,
            dispatch_even=(not args.batching_uneven),
            global_batch_size=args.batch_size,
            max_bucket_size=args.bucket_size,
            local_rank=args.gpu,
            shuffle=False,
        )
        # NOTE: global batch size info is not required for evaluation.
        valloader = DataLoader(
            val_set,
            batch_sampler=val_sampler,
            num_workers=args.workers,
            collate_fn=collate_fn,
        )

        self.train_sampler = tr_sampler
        self.trainloader = trainloader
        self.valloader = valloader

        # Initial model
        cfg = coreutils.readjson(args.config)  # type: dict
        cfg["n_utt"] = len(tr_set)
        self.model = func_build_model(cfg, args)

        coreutils.distprint(
            "> model built. # of params: {:.2f} M".format(
                coreutils.count_parameters(self.model) / 1e6
            ),
            args.gpu,
        )

        # get GPU info and create readme.md
        # NOTE: the following function requires the allreduce OP, so don't put it inside the `if...:` block
        gpu_info = coreutils.gather_all_gpu_info(args.gpu)
        if args.rank == 0 and not args.debug:
            coreutils.gen_readme(
                os.path.join(args.dir, F_TRAINING_INFO),
                model=self.model,
                gpu_info=gpu_info,
            )

        # hook the function
        self.train = train if func_train is None else func_train
        self.evaluate = evaluate if func_eval is None else func_eval

        # Initial specaug module
        if "specaug" not in cfg:
            specaug = None
        else:
            specaug = SpecAug(**cfg["specaug"]).to(f"cuda:{args.gpu}")
        self.specaug = specaug

        # Initial scheduler and optimizer
        assert "scheduler" in cfg
        self.scheduler = build_scheduler(cfg["scheduler"], self.model.parameters())

        # Initialize the grad scaler
        self.scaler = GradScaler(enabled=args.amp)

        self.rank = args.rank  # type: int
        self.DEBUG = args.debug  # type: bool
        self.epoch = 1  # type: int
        self.step = 0  # type: int
        # used to resume from checkpoint
        self.step_by_last_epoch = 0  # type: int

        if not (args.resume is None or args.init_model is None):
            coreutils.distprint(
                "WARNING: you specify both --resume and --init-model, "
                "but --init-model will be ignored.",
                args.rank,
            )

        if args.resume is not None:
            coreutils.distprint(f"> resume from: {args.resume}", args.gpu)
            checkpoint = torch.load(
                args.resume, map_location=f"cuda:{args.gpu}"
            )  # type: OrderedDict
            self.load(checkpoint)
            del checkpoint
        elif args.init_model is not None:
            coreutils.distprint(f"> initialize model from: {args.init_model}", args.gpu)
            checkpoint = torch.load(
                args.init_model, map_location=f"cuda:{args.gpu}"
            )  # type: OrderedDict

            try:
                self.model.load_state_dict(checkpoint["model"])
            except RuntimeError as re:
                if "Error(s) in loading state_dict" in str(re):
                    self.model.load_state_dict(
                        coreutils.translate_prev_checkpoint(checkpoint["model"])
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
            header=f"created by {user} at {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
        )

        # Initialize the tensorboard
        if args.rank == 0:
            print(
                "> use following command to monitor training:\n"
                f"  tensorboard --logdir {args._logdir}"
            )
            self.writer = SummaryWriter(
                os.path.join(args._logdir, "{0:%Y%m%d-%H%M%S/}".format(datetime.now()))
            )
        else:
            self.writer = None

    def run(self, args: argparse.Namespace):
        coreutils.check_parser(args, ["_checkdir", "rank", "gpu", "dir"])
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
                    args.gpu,
                )
            
            for _ in self.train(self.trainloader, args, self):
                self.model.eval()
                metrics = self.evaluate(self.valloader, args, self)
                self.model.train()
                if isinstance(metrics, tuple):
                    # defaultly use the first one to evaluate
                    metrics = metrics[0]

                state = self.scheduler.step(metrics)
                checkpoint = os.path.join(
                    args._checkdir, f"checkpoint.{self.epoch:03}e{self.step}s.pt"
                )
                # inside self.save(), there is an all_reduce OP, don't put it in rank==0 block.
                self.save(checkpoint)
                if self.rank == 0:
                    self.cm.appendinfo(
                        self.epoch,
                        self.step,
                        metrics,
                        self.scheduler.lr_cur,
                        checkpoint,
                    )

                coreutils.distprint(
                    f"Epoch: {self.epoch:<3} | Step: {self.step} | Eval metric: {metrics:.3e} | LR: {self.scheduler.lr_cur:.3e}",
                    args.gpu,
                )

                if state == State.TERMINATED:
                    # backup the last checkpoint
                    if self.rank == 0 and not self.DEBUG:
                        shutil.copyfile(
                            checkpoint, os.path.join(args._checkdir, "checkpoint.pt")
                        )
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

    def save(
        self,
        name: str,
        PATH: str = "",
        extra_states: Optional[Union[Dict, OrderedDict]] = None,
    ) -> str:
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

        if name[-3:] != ".pt":
            name += ".pt"
        PATH = os.path.join(PATH, name)
        states = OrderedDict(
            {
                "model": self.model.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "scaler": self.scaler.state_dict(),
                "step": self.step,
                "step_by_last_epoch": self.step_by_last_epoch,
            }
        )
        if extra_states is not None:
            states.update(extra_states)
        torch.save(states, PATH)
        return PATH

    def load(self, checkpoint: OrderedDict, return_state: bool = False):
        r"Load checkpoint."

        dist.barrier()
        try:
            self.model.load_state_dict(checkpoint["model"])
        except RuntimeError as re:
            if "Error(s) in loading state_dict" in str(re):
                self.model.load_state_dict(
                    coreutils.translate_prev_checkpoint(checkpoint["model"])
                )
            else:
                raise RuntimeError(str(re))

        self.scheduler.load_state_dict(checkpoint["scheduler"])

        # FIXME: This is for old ver. compatible.
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.step_by_last_epoch = checkpoint.get("step_by_last_epoch", self.step)

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


def train(
    trainloader: data.ReadBatchDataLoader,
    args: argparse.Namespace,
    manager: Manager,
    hook_func: Callable = None,
):
    """
    The default train function.

    Args:
        trainloader (Dataloader)
        args (Namespace) : configurations
        manager (Manager) : the manager for pipeline control
        _trainer_hook (optional, callable function) : custom hook function, check source code for usage.
    """

    def _go_step(minibatch) -> Tuple[torch.Tensor, int]:
        # minibatch: feats, feat_lens, labels, label_lens, ...
        minibatch = list(minibatch)
        feats, feat_lens = minibatch[:2]
        if manager.specaug is not None:
            feats = feats.cuda(args.gpu, non_blocking=True)
            feats, feat_lens = manager.specaug(feats, feat_lens)
            minibatch[:2] = [feats, feat_lens]

        with autocast(enabled=use_amp):
            if hook_func is None:
                loss, g2p_loss, s2p_loss, p2g_loss, acc_rate = model(*minibatch)
            else:
                # you could custom model forward, tracks logging and metric calculation in the hook
                loss = hook_func(manager, model, args, i + 1, minibatch)
            if isinstance(loss, tuple):
                loss = loss[0]

            raw_loss = loss.detach().clone()
            # divide loss with fold since we want the gradients to be divided by fold
            loss /= fold

        loss.data = loss.detach() * (feats.size(0) * world_size / trainloader.get_bs())
        # acc_rate = acc_rate * (feats.size(0) * world_size / trainloader.get_bs())
        scaler.scale(loss).backward()

        # return for logging
        return raw_loss, feats.size(0), g2p_loss, s2p_loss, p2g_loss, acc_rate

    coreutils.check_parser(
        args,
        [
            "grad_accum_fold",
            "n_steps",
            "verbose",
            "check_freq",
            "rank",
            "gpu",
            "debug",
            "amp",
            "grad_norm",
        ],
    )

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
    accum_loss = 0.0
    accum_acc_rate = 0.0
    accum_g2p_loss = 0.0
    accum_s2p_loss = 0.0
    accum_p2g_loss = 0.0
    n_batch = 0
    t_data = 0.0
    t_last_step = time.time()
    t_last_batch = time.time()
    cnt_step_update = 0

    def get_progress_bar():
        return tqdm(
            desc=f"Epoch: {manager.epoch} | train",
            unit="batch",
            total=(args.n_steps if args.check_freq == -1 else args.check_freq),
            disable=(args.gpu != 0 or args.verbose),
            leave=False,
        )

    # when check_freq > epoch size, the progress bar would display in mistake.
    p_bar = get_progress_bar()
    for i, minibatch in enumerate(trainloader):
        # since the gradient fold could be > 1, we need to accumulate the time
        if args.verbose:
            t_data += time.time() - t_last_batch

        # skip steps when resuming from stop training
        if cnt_step_update + manager.step_by_last_epoch < manager.step:
            if fold == 1 or (i + 1) % fold == 0:
                cnt_step_update += 1
                p_bar.update()
                if args.verbose and args.gpu == 0:
                    sys.stderr.write(
                        f"\rIn skipping steps: {cnt_step_update + manager.step_by_last_epoch}/{manager.step}"
                    )
                    sys.stderr.flush()
            continue

        # update every fold times and drop the last few batches (number of which <= fold)
        if fold == 1 or (i + 1) % fold == 0:
            local_loss, local_bs, g2p_loss, s2p_loss, p2g_loss, acc_rate = _go_step(minibatch)
            accum_loss += local_loss * local_bs
            accum_acc_rate += acc_rate * local_bs
            accum_g2p_loss += g2p_loss * local_bs
            accum_s2p_loss += s2p_loss * local_bs
            accum_p2g_loss += p2g_loss * local_bs
            n_batch += local_bs

            if grad_norm > 0.0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_norm, error_if_nonfinite=False
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            manager.step += 1
            scheduler.update_lr_step(manager.step)
            cnt_step_update += 1
            p_bar.update()

            # measure accuracy and record loss; item() can sync gpu & cpu.
            item_loss = (accum_loss / n_batch).item()
            item_acc_rate = (accum_acc_rate / n_batch).item()
            item_g2p_loss = (accum_g2p_loss / n_batch).item()
            item_s2p_loss = (accum_s2p_loss / n_batch).item()
            item_p2g_loss = (accum_p2g_loss / n_batch).item()

            # update tensorboard
            if args.rank == 0:
                manager.writer.add_scalar("train_loss/g2p", item_g2p_loss, manager.step)
                manager.writer.add_scalar("train_loss/s2p", item_s2p_loss, manager.step)
                manager.writer.add_scalar("train_loss/p2g", item_p2g_loss, manager.step)
                manager.writer.add_scalar("accept_rate", item_acc_rate, manager.step)
                manager.writer.add_scalar("lr", scheduler.lr_cur, manager.step)

            if args.verbose:
                coreutils.distprint(
                    f"[{manager.epoch} - {cnt_step_update}/{args.n_steps}] | data {t_data:6.3f} | time {time.time()-t_last_step:6.3f} | "
                    f"loss {item_loss:.2e} | lr {scheduler.lr_cur:.2e}",
                    args.gpu,
                )
                t_data = 0.0
                t_last_step = time.time()

            if args.check_freq != -1 and (manager.step % args.check_freq) == 0:
                p_bar.close()
                yield None
                p_bar = get_progress_bar()

            # reset accumulated loss
            accum_loss = 0.0
            accum_acc_rate = 0.0
            accum_g2p_loss = 0.0
            accum_s2p_loss = 0.0
            accum_p2g_loss = 0.0
            n_batch = 0
        else:
            # gradient accumulation w/o sync
            with model.no_sync():
                local_loss, local_bs, g2p_loss, s2p_loss, p2g_loss, acc_rate = _go_step(minibatch)
                accum_loss += local_loss * local_bs
                accum_acc_rate += acc_rate * local_bs
                accum_g2p_loss += g2p_loss * local_bs
                accum_s2p_loss += s2p_loss * local_bs
                accum_p2g_loss += p2g_loss * local_bs
                n_batch += local_bs

        if args.verbose:
            t_last_batch = time.time()

    manager.step_by_last_epoch += cnt_step_update
    # update n_steps, since we don't know how many steps there are with large dataset mode.
    args.n_steps = cnt_step_update
    p_bar.close()
    if args.check_freq == -1:
        yield
    return


@torch.no_grad()
def evaluate(
    testloader: DataLoader, args: argparse.Namespace, manager: Manager
) -> float:
    model = manager.model
    cnt_seq = 0
    total_s2p_loss = 0.0
    total_p2g_loss = 0.0
    total_per = 0.0
    total_wer = 0.0
    total_NLL = 0.0

    for minibatch in tqdm(
        testloader,
        desc=f"Epoch: {manager.epoch} | eval",
        unit="batch",
        disable=(args.gpu != 0),
        leave=False,
    ):
        """
        Suppose the loss is reduced by mean
        """
        with autocast(enabled=args.amp):
            s2p_loss, p2g_loss, per, wer, NLL = model(*minibatch)

        feats = minibatch[0]
        cnt_seq += feats.size(0)
        total_s2p_loss += s2p_loss.float() * feats.size(0)
        total_p2g_loss += p2g_loss.float() * feats.size(0)
        total_per += per.float().cuda() * feats.size(0)
        total_wer += wer.float().cuda() * feats.size(0)
        total_NLL += NLL.float().cuda() * feats.size(0)

    cnt_seq = total_s2p_loss.new_tensor(cnt_seq)

    # sync info for loggin and further state control
    # NOTE: this sync is required.
    dist.all_reduce(total_s2p_loss, dist.ReduceOp.SUM)
    dist.all_reduce(total_p2g_loss, dist.ReduceOp.SUM)
    dist.all_reduce(total_per, dist.ReduceOp.SUM)
    dist.all_reduce(total_wer, dist.ReduceOp.SUM)
    dist.all_reduce(total_NLL, dist.ReduceOp.SUM)
    dist.all_reduce(cnt_seq, dist.ReduceOp.SUM)
    avg_s2p_loss = (total_s2p_loss / cnt_seq).item()
    avg_p2g_loss = (total_p2g_loss / cnt_seq).item()
    avg_per = (total_per / cnt_seq).item()
    avg_wer = (total_wer / cnt_seq).item()
    avg_NLL = (total_NLL / cnt_seq).item()

    if args.rank == 0:
        manager.writer.add_scalar("dev_loss/NLL", avg_NLL, manager.step)
        manager.writer.add_scalar("dev_loss/s2p", avg_s2p_loss, manager.step)
        manager.writer.add_scalar("dev_loss/p2g", avg_p2g_loss, manager.step)
        manager.writer.add_scalar("dev_PER/s2p", avg_per, manager.step)
        manager.writer.add_scalar("dev_WER/p2g", avg_wer, manager.step)
    return avg_wer


class CheckManager:
    def __init__(self, f_checklist: str, header: str = None) -> None:
        # the checkpoint locations would be used for identification
        """Example
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
        """
        self._checks = OrderedDict()  # type: OrderedDict[str, Dict]
        self._f_checklist = f_checklist

        if header is None:
            header = ""

        if os.path.exists(f_checklist):
            # ignore the new header in case overwritten or duplicated.
            self.getcontent()
        else:
            header = "\n".join(
                "# " + x
                for x in [
                    "Use '#' in a new line to identify a comment",
                    "Field definition:",
                    "    No.epoch No.step metric(loss) LR pathtocheckpoint ...(any append info is ok);",
                    "    the float numbers are saved via (1.0).hex(), use float.fromhex('...') to get original data;",
                    "    the No.step is also saved as hex via hex(123), use int('...', 16) to get original data.",
                    "Header info:",
                    " " * 4 + header.replace("\n", " "),
                ]
            )
            with open(f_checklist, "w") as fit:
                fit.write(header)

    @property
    def content(self):
        return self._checks

    def getcontent(self):
        assert os.path.isfile(self._f_checklist)

        with open(self._f_checklist, "r") as fit:
            for line in fit:
                line = line.strip()
                if line[0] == "#" or line == "":
                    # skip the comments
                    continue
                contents = line.split()
                assert len(contents) >= 5
                n_epoch, n_step, metric, lr, f_check = contents[:5]
                self._checks.update(
                    {
                        f_check: {
                            "epoch": int(n_epoch),
                            "step": int(n_step, 16),
                            "metric": float.fromhex(metric),
                            "lr": float.fromhex(lr),
                            "extra": contents[5:],
                        }
                    }
                )

    def appendinfo(
        self, n_epoch: int, n_step: int, metric: float, lr: float, f_check: str, *args
    ):
        self._checks.update(
            {
                f_check: {
                    "epoch": n_epoch,
                    "step": n_step,
                    "metric": metric,
                    "lr": lr,
                    "extra": list(args),
                }
            }
        )
        orin_text = open(self._f_checklist, "r").read()
        try:
            with open(self._f_checklist, "a") as fot:
                fot.write("\n")
                fot.write(
                    (" " * 4).join(
                        [
                            f"{n_epoch:04}",
                            f"{n_step:#010x}",
                            f"{float(metric).hex()}",
                            f"{float(lr).hex()}",
                            f_check,
                        ]
                        + [str(x) for x in args]
                    )
                )
        except Exception as err:
            with open(self._f_checklist, "w") as fot:
                fot.write(orin_text)
            raise RuntimeError(str(err))
