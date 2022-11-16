"""
Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
          2021-2022 Tsinghua University, Author: Huahuan Zheng
Apache 2.0.
This script shows the implementation of CRF loss function.
"""

import os
from typing import Union, List

import torch
from torch.autograd import Function
from torch.nn import Module

import ctc_crf._C as core
from pkg_resources import get_distribution

__version__ = get_distribution('ctc_crf').version


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, "shouldn't require grads"


class _WARP_CTC_GPU(Function):
    @staticmethod
    def forward(ctx, logits, labels, input_lengths, label_lengths, size_average=True):
        logits = logits.contiguous()

        batch_size = logits.size(0)

        costs_ctc = torch.zeros(logits.size(0))
        act = torch.transpose(logits, 0, 1).contiguous()
        grad_ctc = torch.zeros(act.size()).type_as(logits)

        core.gpu_ctc(act, grad_ctc, labels, label_lengths,
                     input_lengths, logits.size(0), costs_ctc, 0)

        grad_ctc = torch.transpose(grad_ctc, 0, 1)
        costs_ctc = costs_ctc.to(logits.get_device())

        grad_all = - grad_ctc
        costs_all = - costs_ctc
        costs = torch.FloatTensor([costs_all.sum()]).to(logits.get_device())

        if size_average:
            grad_all = grad_all / batch_size
            costs = costs / batch_size

        ctx.grads = grad_all
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grads * grad_output.to(ctx.grads.device), None, None, None, None, None, None


class _CTC_CRF(Function):
    @staticmethod
    def forward(ctx, logits, labels, input_lengths, label_lengths, lamb=0.1, size_average=True):
        logits = logits.contiguous()

        batch_size = logits.size(0)
        costs_alpha_den = torch.zeros(logits.size(0)).type_as(logits)
        costs_beta_den = torch.zeros(logits.size(0)).type_as(logits)

        grad_den = torch.zeros(logits.size()).type_as(logits)

        costs_ctc = torch.zeros(logits.size(0))
        act = torch.transpose(logits, 0, 1).contiguous()
        grad_ctc = torch.zeros(act.size()).type_as(logits)

        core.gpu_ctc(act, grad_ctc, labels, label_lengths,
                     input_lengths, logits.size(0), costs_ctc, 0)
        core.gpu_den(logits, grad_den, input_lengths.cuda(),
                     costs_alpha_den, costs_beta_den)

        grad_ctc = torch.transpose(grad_ctc, 0, 1)
        costs_ctc = costs_ctc.to(logits.get_device())

        grad_all = grad_den - (1 + lamb) * grad_ctc
        costs_all = costs_alpha_den - (1 + lamb) * costs_ctc
        costs = torch.FloatTensor([costs_all.sum()]).to(logits.get_device())

        if size_average:
            grad_all = grad_all / batch_size
            costs = costs / batch_size

        ctx.grads = grad_all
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grads * grad_output.to(ctx.grads.device), None, None, None, None, None, None


class CTC_CRF_LOSS(Module):
    def __init__(self, lamb: float = 0.1, size_average: bool = True):
        """
        lamb (float): weight for auxiliary CTC loss, final loss = lamb * loss_ctc + loss_crf
        size_average (bool): whether to do average over batch size dimension.
        """
        super(CTC_CRF_LOSS, self).__init__()
        self.ctc_crf = _CTC_CRF.apply
        self.lamb = lamb
        self.size_average = size_average

    def forward(self, logits, labels, lx, ly) -> torch.FloatTensor:
        """
        logits (torch.FloatTensor): size (N, T, V), on GPU device.
        labels (torch.IntTensor)  : size (\sum {Ui}, ) labels are flattened without paddings,
            and should be placed on CPU.
        lx (torch.IntTensor) : size (N, ), on CPU
        ly (torch.IntTensor) : size (N, ), on CPU
        """
        assert len(labels.size()) == 1
        assert logits.dtype == torch.float, f"expect logits to be torch.float object, instead: {logits.dtype}"
        assert labels.dtype == torch.int, f"expect labels to be torch.int object, instead: {labels.dtype}"
        assert lx.dtype == torch.int, f"expect lx to be torch.int object, instead: {lx.dtype}"
        assert ly.dtype == torch.int, f"expect ly to be torch.int object, instead: {ly.dtype}"

        _assert_no_grad(labels)
        _assert_no_grad(lx)
        _assert_no_grad(ly)
        return self.ctc_crf(logits, labels, lx, ly, self.lamb, self.size_average)


class WARP_CTC_LOSS(Module):
    """
    This impl of CTC loss is not recommended.
    Please consider using torch.nn.CTCLoss
    """

    def __init__(self, size_average=True):
        super(WARP_CTC_LOSS, self).__init__()
        self.ctc = _WARP_CTC_GPU.apply
        self.size_average = size_average

    def forward(self, logits, labels, input_lengths, label_lengths):
        assert len(labels.size()) == 1
        _assert_no_grad(labels)
        _assert_no_grad(input_lengths)
        _assert_no_grad(label_lengths)
        return self.ctc(logits, labels, input_lengths, label_lengths, self.size_average)


class CRFContext:
    def __init__(self, den_lm: str, gpus: Union[int, List[int]]) -> None:
        """
        den_lm (str): path to the denominator LM. For how to prepare a den_lm, please refer to
            cat/utils/tool/prep_den_lm.sh
        gpus   (int, List[int]): Specify which GPU to be used.
        """
        if not os.path.isfile(den_lm):
            raise RuntimeError(
                f"Denominator LM model location is invalid: {den_lm}.")

        if isinstance(gpus, int):
            gpus = [gpus]
        nprocs = torch.cuda.device_count()
        if not all([i >= 0 and i < nprocs for i in gpus]):
            raise RuntimeError(
                f"Available GPU={nprocs}, invalid GPU ids: {gpus}.")

        self._gpus = torch.IntTensor(gpus)
        core.init_env(den_lm, self._gpus)

    def __del__(self):
        if hasattr(self, '_gpus'):
            core.release_env(self._gpus)
            del self._gpus
