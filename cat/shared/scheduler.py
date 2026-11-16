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


def build_scheduler(
    cfg: dict, params: Iterable[torch.nn.parameter.Parameter]
) -> "Scheduler":
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
    assert (
        "type" in cfg
    ), "you should specify the 'type' field in scheduler configuration."
    cls_scdl = cfg["type"]  # type: str
    assert isinstance(cfg["type"], str)
    assert (
        cls_scdl.isidentifier()
    ), f"invalid type name: {cls_scdl}, not even valid as an identifier."
    cls_scdl = eval(cls_scdl)  # type: Scheduler
    return cls_scdl(
        optimizer=_build_optim(cfg=cfg["optimizer"], params=params),
        **cfg.get("kwargs", {}),
    )


def _build_optim(
    cfg: dict, params: Iterable[torch.nn.parameter.Parameter]
) -> Union[torch.optim.Optimizer, ZeroRedundancyOptimizer]:
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

    assert "type" in cfg
    cls_optim = cfg["type"]
    kwarg_optim = cfg.get("kwargs", {})

    assert isinstance(cls_optim, str)
    # type: torch.optim.Optimizer
    cls_optim = getattr(torch.optim, cls_optim)
    if cfg.get("zeroredundancy", False):
        return ZeroRedundancyOptimizer(
            params=params, optimizer_class=cls_optim, **kwarg_optim
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
        self.best_metric = float("-inf") if reverse else float("inf")

    @property
    def lr_cur(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _adjust_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _is_improved(self, metric: float) -> bool:
        return self._reverse ^ (metric < self.best_metric)

    def state_dict(self):
        output = OrderedDict()
        for name, attr in vars(self).items():
            if name == "optimizer":
                output["optimizer"] = attr.state_dict()
            else:
                output[name] = attr
        return output

    def load_state_dict(self, ckpt: OrderedDict, optim_only: bool = False):
        """Load state dict from checkpoint.

        By setting `optim_only`, it allows to update the configurations of scheduler
        """
        if optim_only:
            self.optimizer.load_state_dict(ckpt["optimizer"])
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
        min_step: int,  # number of the minimum steps
        stop_lr: float = 1e-5,  # LR to encounter the early stop
        # number of look-forward iterations after encountering the worse metric
        n_tol: int = 0,
        gamma: float = 0.1,  # decay factor, LR(n+1) = gamma*LR(n)
        reverse: bool = False,
    ):
        super().__init__(optimizer, reverse)

        assert stop_lr > 0.0
        assert gamma > 0.0
        assert min_step >= 0
        assert not ((stop_lr < self.lr_cur) ^ (gamma < 1.0)), (
            f"in the configure: init-lr={self.init_lr:.2e} "
            f"stop-lr={stop_lr:.2e} gamma={gamma:.2e}\n"
            "the training will never terminated."
        )
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
                if self._check_hit_stop(self.lr_cur * self.gamma):
                    return State.TERMINATED
                else:
                    self._adjust_lr(self.lr_cur * self.gamma)
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
        reverse: bool = False,
    ):
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
        ref_lr: float = 0.0,
        stop_lr: float = 1e-5,
        n_tol: int = 1,
        gamma: float = 0.1,
        reverse: bool = False,
    ):
        super().__init__(
            optimizer,
            min_step=warmup_step,
            stop_lr=stop_lr,
            n_tol=n_tol,
            gamma=gamma,
            reverse=reverse,
        )
        if ref_lr == 0.0:
            ref_lr = self.lr_cur

        assert batch_size > 0
        assert warmup_step > 0
        assert ref_bs > 0
        assert ref_lr > 0

        if self.lr_cur != ref_lr:
            print(
                "Warning: the learning set in optimizer and `refer_lr` are different."
            )
            self._adjust_lr(ref_lr)

        max_lr = max(batch_size / ref_bs * ref_lr, ref_lr)
        self.lr_addon = (max_lr - self.lr_cur) / warmup_step

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
        reverse: bool = False,
    ):
        super().__init__(optimizer, stop_step, reverse)

        assert dim_model > 0
        assert warmup_step > 0
        assert peak_factor > 0.0
        self.init_lr = peak_factor / math.sqrt(dim_model)
        self._den_in_warmup = 1.0 / math.sqrt(warmup_step) / warmup_step
        self.update_lr_step(1)

    def update_lr_step(self, n_step: int):
        super().update_lr_step(n_step)
        lr = self.init_lr * min(1.0 / math.sqrt(n_step), n_step * self._den_in_warmup)
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
        reverse: bool = False,
    ):
        assert dim_model > 0
        assert warmup_step > 0
        assert peak_factor > 0.0
        if min_step == -1:
            min_step = warmup_step
        assert min_step >= warmup_step
        super().__init__(optimizer, min_step, stop_lr, n_tol, gamma, reverse)

        self.ref_lr = peak_factor / math.sqrt(dim_model)
        self._den_in_warmup = 1.0 / math.sqrt(warmup_step) / warmup_step
        self.update_lr_step(1)

    def update_lr_step(self, n_step: int):
        super().update_lr_step(n_step)
        lr = self.ref_lr * min(1.0 / math.sqrt(n_step), n_step * self._den_in_warmup)
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
        reverse: bool = False,
    ):
        super().__init__(optimizer, stop_step, reverse)

        assert stop_lr > 0.0 and stop_lr < self.lr_cur
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
        lr = self.lr_init * (self.decay**self.iter_cur)
        self._adjust_lr(lr)


class SchedulerCosineAnnealing(SchedulerFixedStop):
    """Annealing the LR with cosine function (and period) by steps."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        min_lr: float,
        stop_step: int,
        period: int = 0,
        decay_factor: float = 1.0,
        reverse: bool = False,
    ):
        super().__init__(optimizer, stop_step, reverse)

        assert min_lr > 0.0
        assert period >= 0
        assert decay_factor > 0.0 and decay_factor <= 1.0
        if period == 0:
            period = stop_step

        self.period = period
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self._ref_lr = self.lr_cur

    def update_lr_step(self, n_step: int):
        super().update_lr_step(n_step)
        max_lr = self._ref_lr * self.decay_factor ** ((n_step - 1) // self.period)
        self._adjust_lr(
            self.min_lr
            + 0.5
            * (max_lr - self.min_lr)
            * (1 + math.cos(((n_step - 1) % self.period) / self.period * math.pi))
        )


class SchedulerCosineAnnealingWithWarmup(Scheduler):
    """
    Cosine annealing with a linear warmup phase.

    Design goals:
        1) The cosine timeline starts after warmup instead of reusing the global
           training step. This makes the first post-warmup cosine step start
           from the peak LR smoothly, rather than jumping to a later cosine
           position.
        2) The scheduler only runs a single cosine decay after warmup. If the
           configured training length is longer than one cosine period, the LR
           stays at `min_lr` instead of silently entering a new cosine cycle.

    Notes:
        - `start_lr` controls the starting LR of the warmup phase.
        - `min_lr` controls the floor LR of the cosine decay phase.
        - `decay_factor` is kept for configuration compatibility with
          `SchedulerCosineAnnealing`, but it is intentionally not used here
          because this scheduler disables cosine restarts by design.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        min_lr: float,
        stop_step: int,
        start_lr: Optional[float] = None,
        period: int = 0,
        decay_factor: float = 1.0,
        warmup_steps: int = 0,
        reverse: bool = False,
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer): The optimizer instance.
            min_lr (float): The minimum LR reached after cosine decay.
            stop_step (int): Total number of training steps.
            start_lr (Optional[float]): Warmup starting LR. If not provided,
                the warmup starts from `min_lr` to preserve the old behavior.
            period (int): Number of steps used by the cosine decay stage.
                If set to 0, the scheduler uses all post-warmup steps.
                If set larger than the post-warmup steps, it will be clipped
                so that the cosine phase fits within the training budget.
            decay_factor (float): Reserved only for config compatibility.
            warmup_steps (int): Number of linear warmup steps.
            reverse (bool): Reverse the optimization order of evaluation metric.
        """
        assert warmup_steps >= 0
        assert warmup_steps < stop_step, (
            f"warmup_steps ({warmup_steps}) must be smaller than stop_step "
            f"({stop_step}) so that the cosine phase has at least one step."
        )

        super().__init__(optimizer, reverse)

        assert min_lr > 0.0
        assert stop_step > 0
        assert decay_factor > 0.0 and decay_factor <= 1.0

        # Keep the same fixed-stop attributes as SchedulerFixedStop so that
        # state dicts and evaluation behavior remain compatible.
        self.stop_step = int(stop_step)
        self._in_stop_step = True

        # Keep the same cosine-related attributes as SchedulerCosineAnnealing
        # so that existing checkpoints can still be loaded without schema drift.
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self._ref_lr = self.lr_cur

        self.warmup_steps = warmup_steps
        self.start_lr = min_lr if start_lr is None else start_lr
        assert self.start_lr > 0.0

        # `cosine_steps` is the effective training budget after warmup. This is
        # the correct time axis for the cosine phase.
        self.cosine_steps = self.stop_step - self.warmup_steps

        # We only allow a single cosine decay here. When `period` is shorter
        # than the remaining training length, the LR decays to `min_lr` and
        # then stays there, avoiding accidental restarts.
        if period == 0:
            self.period = self.cosine_steps
        else:
            assert period > 0
            self.period = min(period, self.cosine_steps)

        # The current training managers call `scheduler.update_lr_step(...)`
        # after `optimizer.step()`. Therefore the scheduler needs to preload the
        # LR for the very first optimizer update here, otherwise the first
        # update would incorrectly use the optimizer's peak LR directly.
        self._adjust_lr(self._get_lr_by_train_step(1))

    def _get_lr_by_train_step(self, step: int) -> float:
        """
        Compute the LR that should be used for a given optimizer update step.

        Args:
            step (int): 1-based optimizer update index.

        Returns:
            float: The LR assigned to that optimizer update.
        """
        assert step >= 1

        step = min(step, self.stop_step)

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            # Linear warmup:
            #   step == 1              -> close to `start_lr`
            #   step == warmup_steps   -> exactly `_ref_lr`
            warmup_ratio = step / self.warmup_steps
            return self.start_lr + (self._ref_lr - self.start_lr) * warmup_ratio

        # Shift the cosine timeline so that the first post-warmup step uses
        # cosine step 1, which yields the peak LR and therefore connects
        # smoothly to the end of warmup.
        cosine_step = max(step - self.warmup_steps, 1)

        # Clamp the cosine step to a single period. This prevents an
        # accidentally long training run from re-entering a new cosine
        # cycle at the end.
        cosine_step = min(cosine_step, self.period)

        # For a single-cycle scheduler we want:
        #   cosine_step == 1           -> lr == _ref_lr
        #   cosine_step == self.period -> lr == min_lr
        # Therefore we map the discrete cosine steps to [0, pi] with
        # `self.period - 1` intervals instead of the restart-style
        # denominator used by the base scheduler.
        if self.period == 1:
            return self.min_lr

        cosine_progress = (cosine_step - 1) / (self.period - 1)
        return self.min_lr + 0.5 * (self._ref_lr - self.min_lr) * (
            1 + math.cos(cosine_progress * math.pi)
        )

    def update_lr_step(self, n_step: int):
        """
        Update LR by training step (minibatch level).

        PyTorch-style behavior:
            - The manager calls this function after each optimizer update, so
              `n_step` is treated as the number of finished updates.
            - This function therefore prepares the LR for the next optimizer
              update, i.e. for training step `n_step + 1`.
            - During warmup, LR increases linearly from `start_lr` to `_ref_lr`.
            - After warmup, the cosine phase is driven by the post-warmup
              relative step instead of the global step.
            - Once the cosine phase reaches its final step, LR is clamped at
              `min_lr` instead of starting a new cosine cycle.
        """
        # Mirror SchedulerFixedStop bookkeeping locally because this scheduler
        # now inherits from the base Scheduler directly.
        if self._in_stop_step and n_step >= self.stop_step:
            self._in_stop_step = False

        # `n_step` is the number of finished optimizer updates in the current
        # managers, so the LR we set here will be consumed by the next update.
        next_step = min(max(n_step + 1, 1), self.stop_step)
        self._adjust_lr(self._get_lr_by_train_step(next_step))

    def step(self, metric: float, *args, **kwargs) -> State:
        """
        Return the fixed-stop scheduler state during evaluation.

        This keeps the same behavior as `SchedulerFixedStop`: before the
        training reaches `stop_step`, evaluation reports whether the metric is
        improved; after `stop_step`, the scheduler asks training to terminate.
        """
        if self._in_stop_step:
            if self._is_improved(metric):
                return State.IMPROVED
            else:
                return State.CONTINUE
        else:
            return State.TERMINATED


class SchedulerCosineAnnealingWithWarmupFix(SchedulerCosineAnnealingWithWarmup):
    """
    Backward-compatible alias for the fixed warmup-cosine scheduler.

    Some older experiment notes use the `WithWarmupFix` name. Keeping this
    alias avoids breaking those configurations while reusing the corrected
    implementation above.
    """

    pass
