# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""Optimizer scheduler impl"""

import math
import torch
from typing import *
from enum import Enum
from collections import OrderedDict

from torch.distributed.optim import ZeroRedundancyOptimizer


def build_scheduler(cfg: dict, params: Iterable[torch.nn.parameter.Parameter]) -> "Scheduler":
    """Build the scheduler from configurations and parameter list.
    
    Args:
        cfg (dict): configuration of scheduler, including the configuration for optimizer
            {
                "type": "NameOfScheduler",
                "kwargs": {
                    ...
                },
                "optimizer": ... // see _build_optim for the format
            }
        params (iterable[parameter]): parameters to be optimized

    """

    # these assertions are required for avoiding toxic code to be executed in eval()
    assert 'type' in cfg, \
        "you should specify the 'type' field in scheduler configuration."
    cls_scdl = cfg['type']     # type: str
    assert isinstance(cfg['type'], str)
    assert cls_scdl.isidentifier(
    ), f"invalid type name: {cls_scdl}, not even valid as an identifier."
    cls_scdl = eval(cls_scdl)     # type: Scheduler
    return cls_scdl(
        optimizer=_build_optim(
            cfg=cfg['optimizer'],
            params=params
        ),
        **cfg.get('kwargs', {})
    )


def _build_optim(cfg: dict, params: Iterable[torch.nn.parameter.Parameter]) -> Union[torch.optim.Optimizer, ZeroRedundancyOptimizer]:
    """Setup the optimizer.

    Args:
        cfg (dict): in following format
            {
                // name of optimizer, should be an attribute of `torch.optim`, like `Adam`, `SGD`.
                "type": "NameOfOptimizerClass",
                // 'zeroredundancy' is a flag to determinte whether use `ZeroRedundancyOptimizer` or not,
                // ... ref to https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html, 
                // ... which is supported since torch 1.8.0
                "zeroredundancy": true,
                // 'kwargs' contains any keyword arguments can be passed into optimizer initialization.
                // such as lr, momentum (SGD), betas (Adam) ...
                "kwargs": {
                    ...
                }
            }

    Return:
        optimizer (torch.optim.Optimizer | ZeroRedundancyOptimizer)
    """

    assert 'type' in cfg
    cls_optim = cfg['type']
    kwarg_optim = cfg.get('kwargs', {})

    assert isinstance(cls_optim, str)
    # type: torch.optim.Optimizer
    cls_optim = getattr(torch.optim, cls_optim)
    if cfg.get('zeroredundancy', False):
        return ZeroRedundancyOptimizer(
            params=params,
            optimizer_class=cls_optim,
            **kwarg_optim
        )
    else:
        return cls_optim(params, **kwarg_optim)


class State(Enum):
    """Evaluation state.

    meaning:
    IMPROVED (0): continue training for metric is improving
    CONTINUE (1): continue training by the some predefined condition
    TERMINATED (2): stop training.
    """
    IMPROVED = 0
    CONTINUE = 1
    TERMINATED = 2


class Scheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, reverse: bool = False):
        """
        Args:
            optimizer : optimizer
            reverse (bool): reverse the optimizing order of evaluation metric, defaultly is descending order.
        """
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(reverse, bool)

        self.optimizer = optimizer
        self._reverse = reverse
        self.init_lr = self.lr_cur
        self.best_metric = float('-inf') if reverse else float('inf')

    @property
    def lr_cur(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def _adjust_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _is_improved(self, metric: float) -> bool:
        return self._reverse ^ (metric < self.best_metric)

    def state_dict(self):
        output = OrderedDict()
        for name, attr in vars(self).items():
            if name == 'optimizer':
                output['optimizer'] = attr.state_dict()
            else:
                output[name] = attr
        return output

    def load_state_dict(self, ckpt: OrderedDict, optim_only: bool = False):
        """Load state dict from checkpoint.

        By setting `optim_only`, it allows to update the configurations of scheduler
        """
        if optim_only:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            return None

        for name in vars(self).keys():
            if name == "optimizer":
                self.optimizer.load_state_dict(ckpt[name])
            else:
                setattr(self, name, ckpt[name])

    def update_lr_step(self, n_step: int):
        """Method for updating the LR by steps. Defaultly do nothing."""
        return None

    def step(self, metric: float, *args, **kwargs) -> State:
        """Scheduler step.
        Update the scheduler by every evaluation, useful for early-stop.

        Args:
            metric (float): the metric for evaluate the performance

        Returns: 
            state (State)
        """
        raise NotImplementedError


class SchedulerEarlyStop(Scheduler):
    """A scheduler wrapper for early-stop ones.
    
    ## LR updating way:
    
    ### By minibatches/steps:
        Do nothing.
        
    ### By evaluation time:

        if `metric` is improved, return `IMPROVED`

        elif `step < min_step`, return `CONTINUE`

        else

            if `cnt_worse_eval <= n_tol`, return `CONTINUE`

            else

                if `lr_cur > stop_lr`, do `LR *= gamma`, return `CONTINUE`
                
                else return `TERMINATED`
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            min_step: int,              # number of the minimum steps
            stop_lr: float = 1e-5,      # LR to encounter the early stop
            # number of look-forward iterations after encountering the worse metric
            n_tol: int = 0,
            gamma: float = 0.1,         # decay factor, LR(n+1) = gamma*LR(n)
            reverse: bool = False):
        super().__init__(optimizer, reverse)

        assert stop_lr > 0.
        assert gamma > 0.
        assert min_step >= 0
        assert not ((stop_lr < self.lr_cur) ^ (gamma < 1.)), \
            f"in the configure: init-lr={self.init_lr:.2e} " \
            f"stop-lr={stop_lr:.2e} gamma={gamma:.2e}\n" \
            "the training will never terminated."
        assert n_tol >= 0

        self.stop_lr = stop_lr

        self.min_step = min_step
        self._in_min_step = True

        self.n_tol = n_tol
        self._cnt_worse = 0

        self.gamma = gamma

    def _check_hit_stop(self, new_lr: float) -> bool:
        return (self.stop_lr <= new_lr) ^ (self.gamma < 1.0)

    def update_lr_step(self, n_step: int):
        if self._in_min_step and n_step >= self.min_step:
            self._in_min_step = False
        return None

    def step(self, metric: float) -> State:
        if self._is_improved(metric):
            self.best_metric = metric
            return State.IMPROVED
        elif self._in_min_step:
            return State.CONTINUE
        else:
            self._cnt_worse += 1
            if self._cnt_worse > self.n_tol:
                if self._check_hit_stop(self.lr_cur*self.gamma):
                    return State.TERMINATED
                else:
                    self._adjust_lr(self.lr_cur*self.gamma)
                    self._cnt_worse = 0
                    return State.CONTINUE
            else:
                return State.CONTINUE


class SchedulerFixedStop(Scheduler):
    """A scheduler wrapper for ones stopping at fixed iterations.
    
    ## LR updating way:
    
    ### By minibatches/steps:
        do nothing
        
    ### By evaluation time:
        if `step < stop_step`
            if `metric` is improved
                return IMPROVED
            else
                return CONTINUE
        else
            return TERMINATED
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        # the step to terminate the training,
        # note that training won't be terminated hitting stop_step
        # ... unitl next evaluation.
        stop_step: int,
            reverse: bool = False):
        super().__init__(optimizer, reverse)

        assert stop_step > 0
        self.stop_step = int(stop_step)
        self._in_stop_step = True

    def update_lr_step(self, n_step: int):
        if self._in_stop_step and n_step >= self.stop_step:
            self._in_stop_step = False
        return

    def step(self, metric: float, *args, **kwargs) -> State:
        if self._in_stop_step:
            if self._is_improved(metric):
                return State.IMPROVED
            else:
                return State.CONTINUE
        else:
            return State.TERMINATED


class SchedulerEarlyStopWithWarmup(SchedulerEarlyStop):
    """MileStone scheduler with warmup
        
    Combine the linear warmup and mile stone decreasing up

    ## LR updating way:
    
    ### By minibatches/steps:
        if `step < warmup_step`
            linearly increasing the lr to `max_lr`
        
    ### By evaluation time:
        refer to SchedulerEarlyStop
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            batch_size: int,
            warmup_step: int,
            ref_bs: int,
            ref_lr: float = 0.,
            stop_lr: float = 1e-5,
            n_tol: int = 1,
            gamma: float = 0.1,
            reverse: bool = False):

        super().__init__(optimizer, min_step=warmup_step, stop_lr=stop_lr,
                         n_tol=n_tol, gamma=gamma, reverse=reverse)
        if ref_lr == 0.:
            ref_lr = self.lr_cur

        assert batch_size > 0
        assert warmup_step > 0
        assert ref_bs > 0
        assert ref_lr > 0

        if self.lr_cur != ref_lr:
            print("Warning: the learning set in optimizer and `refer_lr` are different.")
            self._adjust_lr(ref_lr)

        max_lr = max(batch_size/ref_bs * ref_lr, ref_lr)
        self.lr_addon = (max_lr-self.lr_cur)/warmup_step

    def update_lr_step(self, n_step: int):
        if self._in_min_step:
            self._adjust_lr(self.lr_cur + self.lr_addon)
            if n_step >= self.min_step:
                self._in_min_step = False
        return None


class SchedulerNoam(SchedulerFixedStop):
    """
    The standard scheduler of "Attention is all you need"

    peak learning rate = peak_factor / sqrt(warmup_step * dim_model)
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            # hidden dimension of model (attention layer in the original paper).
            dim_model: int,
            warmup_step: int,
            stop_step: int,
            peak_factor: float = 1.0,
            reverse: bool = False):
        super().__init__(optimizer, stop_step, reverse)

        assert dim_model > 0
        assert warmup_step > 0
        assert peak_factor > 0.
        self.init_lr = peak_factor / math.sqrt(dim_model)
        self._den_in_warmup = 1./math.sqrt(warmup_step)/warmup_step
        self.update_lr_step(1)

    def update_lr_step(self, n_step: int):
        super().update_lr_step(n_step)
        lr = self.init_lr * \
            min(1./math.sqrt(n_step),  n_step*self._den_in_warmup)
        self._adjust_lr(lr)


class SchedulerNoamEarlyStop(SchedulerEarlyStop):
    """
    Linear warmup by step + decay by step + early stop by iteration

    peak lr = peak_factor / sqrt(dim_model * warmup_step)
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            dim_model: int,
            warmup_step: int,
            peak_factor: float = 1.0,
            stop_lr: float = 0.00001,
            n_tol: int = 0,
            gamma: float = 0.1,
            min_step: Optional[int] = -1,
            reverse: bool = False):
        assert dim_model > 0
        assert warmup_step > 0
        assert peak_factor > 0.
        if min_step == -1:
            min_step = warmup_step
        assert min_step >= warmup_step
        super().__init__(optimizer, min_step, stop_lr, n_tol, gamma, reverse)

        self.ref_lr = peak_factor / math.sqrt(dim_model)
        self._den_in_warmup = 1./math.sqrt(warmup_step)/warmup_step
        self.update_lr_step(1)

    def update_lr_step(self, n_step: int):
        super().update_lr_step(n_step)
        lr = self.ref_lr * \
            min(1./math.sqrt(n_step),  n_step*self._den_in_warmup)
        self._adjust_lr(lr)

    def step(self, metric: float) -> State:
        prev_lr = self.lr_cur
        state = super().step(metric)
        self.ref_lr *= self.lr_cur / prev_lr
        return state


class SchedulerLinearAnnealing(SchedulerFixedStop):
    """
    Linearly annealing the LR by every step.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            min_step: int,
            stop_lr: float,
            stop_step: int,
            reverse: bool = False):
        super().__init__(optimizer, stop_step, reverse)

        assert stop_lr > 0. and stop_lr < self.lr_cur
        assert min_step >= 0
        assert stop_step > min_step

        self.min_step = min_step
        self._in_min_step = True
        self._lr_addon = -(self.lr_cur - stop_lr) / (stop_step - min_step)

    def update_lr_step(self, n_step: int):
        if self._in_min_step:
            if n_step >= self.min_step:
                self._in_min_step = False
        elif self._in_stop_step:
            self._adjust_lr(self.lr_cur + self._lr_addon)
            if n_step >= self.stop_step:
                self._in_stop_step = False

        return None

    def _impl_update_lr_iter(self):
        lr = self.lr_init * (self.decay ** self.iter_cur)
        self._adjust_lr(lr)


class SchedulerCosineAnnealing(SchedulerFixedStop):
    """Annealing the LR with cosine function (and period) by steps."""

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            min_lr: float,
            stop_step: int,
            period: int = 0,
            decay_factor: float = 1.,
            reverse: bool = False):
        super().__init__(optimizer, stop_step, reverse)

        assert min_lr > 0.
        assert period >= 0
        assert decay_factor > 0. and decay_factor <= 1.
        if period == 0:
            period = stop_step

        self.period = period
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self._ref_lr = self.lr_cur

    def update_lr_step(self, n_step: int):
        super().update_lr_step(n_step)
        max_lr = (self._ref_lr * self.decay_factor**((n_step-1)//self.period))
        self._adjust_lr(
            self.min_lr + 0.5 * (max_lr - self.min_lr) * (
                1 + math.cos(((n_step-1) % self.period) /
                             self.period * math.pi)
            )
        )
