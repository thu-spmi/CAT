"""
Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
          2021-2022 Tsinghua University, Author: Huahuan Zheng
Apache 2.0.
This script shows the implementation of the Latent CTC loss function.
"""
import json
from typing import Union, Optional, Sequence, Tuple

import torch
from torch.autograd import Function
from torch.nn import Module

import latent_ctc._C as core
from pkg_resources import get_distribution

__version__ = get_distribution('latent_ctc').version


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, "shouldn't require grads"


class _LATENT_CTC_GPU(Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        labels,
        input_lengths,
        label_lengths,
        major_offsets,
        major_flat_subclasses,
        major_label_offset,
        size_average=True,
        blank_label=0,
    ):
        logits = logits.contiguous()
        batch_size = logits.size(0)

        act = torch.transpose(logits, 0, 1).contiguous()
        grad_ctc = torch.zeros(act.size()).type_as(logits)
        costs_ctc = torch.zeros(batch_size)

        # The latent CUDA kernel is fragile on some CUDA/Hopper builds when a
        # minibatch mixes very different label lengths.  Run each utterance as a
        # B=1 CTC problem; this preserves the exact loss/gradient semantics while
        # avoiding the problematic cross-utterance workspace path.
        if batch_size == 1:
            try:
                core.gpu_latent_ctc(
                    act,
                    grad_ctc,
                    labels,
                    label_lengths,
                    input_lengths,
                    batch_size,
                    costs_ctc,
                    blank_label,
                    major_offsets,
                    major_flat_subclasses,
                    major_label_offset,
                )
            except Exception:
                L = int(label_lengths[0].item())
                T = int(input_lengths[0].item())
                repeat = int((labels[1:] == labels[:-1]).sum().item()) if L > 1 else 0
                S = 2 * L + 1
                print(
                    "[latent_ctc failed B=1]",
                    "T=", T,
                    "L=", L,
                    "repeat=", repeat,
                    "S=", S,
                    "min_label=", int(labels.min().item()) if L > 0 else None,
                    "max_label=", int(labels.max().item()) if L > 0 else None,
                    "num_major=", int((labels >= major_label_offset).sum().item()) if L > 0 else 0,
                    "labels=", labels.tolist(),
                    flush=True,
                )
                raise
        else:
            label_start = 0
            for b in range(batch_size):
                cur_label_len = int(label_lengths[b].item())
                cur_labels = labels[label_start: label_start + cur_label_len].contiguous()
                cur_label_lengths = label_lengths[b: b + 1].contiguous()
                cur_input_lengths = input_lengths[b: b + 1].contiguous()
                cur_cost = torch.zeros(1)
                cur_grad = torch.zeros(act[:, b: b + 1, :].size()).type_as(logits)

                try:
                    core.gpu_latent_ctc(
                        act[:, b: b + 1, :].contiguous(),
                        cur_grad,
                        cur_labels,
                        cur_label_lengths,
                        cur_input_lengths,
                        1,
                        cur_cost,
                        blank_label,
                        major_offsets,
                        major_flat_subclasses,
                        major_label_offset,
                    )
                except Exception:
                    L = int(cur_label_lengths[0].item())
                    T = int(cur_input_lengths[0].item())
                    repeat = int((cur_labels[1:] == cur_labels[:-1]).sum().item()) if L > 1 else 0
                    S = 2 * L + 1
                    print(
                        "[latent_ctc failed]",
                        "batch_index=", b,
                        "T=", T,
                        "L=", L,
                        "repeat=", repeat,
                        "S=", S,
                        "min_label=", int(cur_labels.min().item()) if L > 0 else None,
                        "max_label=", int(cur_labels.max().item()) if L > 0 else None,
                        "num_major=", int((cur_labels >= major_label_offset).sum().item()) if L > 0 else 0,
                        "labels=", cur_labels.tolist(),
                        flush=True,
                    )
                    raise


                grad_ctc[:, b: b + 1, :] = cur_grad
                costs_ctc[b] = cur_cost[0]
                label_start += cur_label_len

        grad_ctc = torch.transpose(grad_ctc, 0, 1)
        costs_ctc = costs_ctc.to(logits.get_device())

        grad_all = -grad_ctc
        costs_all = -costs_ctc
        costs = torch.FloatTensor([costs_all.sum()]).to(logits.get_device())

        if size_average:
            grad_all = grad_all / batch_size
            costs = costs / batch_size

        ctx.grads = grad_all
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        return (
            ctx.grads * grad_output.to(ctx.grads.device),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _build_latent_map_tensors(
    major_to_subclasses: Sequence[Sequence[int]],
) -> Tuple[torch.IntTensor, torch.IntTensor]:
    offsets = [0]
    flat_subclasses = []
    for subclasses in major_to_subclasses:
        if len(subclasses) == 0:
            raise ValueError("each latent major must map to at least one subclass")
        flat_subclasses.extend(int(subclass) for subclass in subclasses)
        offsets.append(len(flat_subclasses))

    return (
        torch.tensor(offsets, dtype=torch.int32),
        torch.tensor(flat_subclasses, dtype=torch.int32),
    )


def _load_latent_map(
    latent_map: Optional[Union[str, dict, Sequence[Sequence[int]]]],
    major_label_offset: Optional[int],
) -> Tuple[torch.IntTensor, torch.IntTensor, int]:
    if latent_map is None:
        if major_label_offset is None:
            major_label_offset = 0
        return (
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
            int(major_label_offset),
        )

    if isinstance(latent_map, str):
        with open(latent_map, "r") as fi:
            latent_map = json.load(fi)

    if isinstance(latent_map, dict):
        major_label_offset = int(
            latent_map.get("major_label_offset", major_label_offset)
        )
        if "major_to_subclasses" in latent_map:
            major_to_subclasses = latent_map["major_to_subclasses"]
        elif "majors" in latent_map:
            major_to_subclasses = [
                subclasses
                for _, subclasses in sorted(
                    latent_map["majors"].items(), key=lambda item: int(item[0])
                )
            ]
        else:
            raise ValueError(
                "latent_map dict must contain 'major_to_subclasses' or 'majors'"
            )
    else:
        major_to_subclasses = latent_map

    if major_label_offset is None:
        raise ValueError("major_label_offset is required for latent CTC")

    major_offsets, major_flat_subclasses = _build_latent_map_tensors(
        major_to_subclasses
    )
    return major_offsets, major_flat_subclasses, int(major_label_offset)


class LATENT_CTC_LOSS(Module):
    """
    GPU CTC loss with optional latent major labels.

    labels may contain either normal subclass ids in [0, V) or major ids encoded
    as major_label_offset + major_index. The network output vocabulary remains
    the subclass vocabulary.
    """

    def __init__(
        self,
        latent_map: Optional[Union[str, dict, Sequence[Sequence[int]]]] = None,
        major_label_offset: Optional[int] = None,
        size_average: bool = True,
        blank_label: int = 0,
    ):
        super(LATENT_CTC_LOSS, self).__init__()
        self.ctc = _LATENT_CTC_GPU.apply
        (
            self.major_offsets,
            self.major_flat_subclasses,
            self.major_label_offset,
        ) = _load_latent_map(latent_map, major_label_offset)
        self.size_average = size_average
        self.blank_label = blank_label

    def forward(self, logits, labels, input_lengths, label_lengths):
        assert len(labels.size()) == 1
        assert logits.dtype == torch.float, f"expect logits to be torch.float object, instead: {logits.dtype}"
        assert labels.dtype == torch.int, f"expect labels to be torch.int object, instead: {labels.dtype}"
        assert input_lengths.dtype == torch.int, f"expect input_lengths to be torch.int object, instead: {input_lengths.dtype}"
        assert label_lengths.dtype == torch.int, f"expect label_lengths to be torch.int object, instead: {label_lengths.dtype}"

        _assert_no_grad(labels)
        _assert_no_grad(input_lengths)
        _assert_no_grad(label_lengths)
        return self.ctc(
            logits,
            labels,
            input_lengths,
            label_lengths,
            self.major_offsets,
            self.major_flat_subclasses,
            self.major_label_offset,
            self.size_average,
            self.blank_label,
        )
