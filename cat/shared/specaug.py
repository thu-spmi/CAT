"""
Copyright to espnet
https://github.com/espnet/espnet/blob/master/espnet2/asr/specaug/specaug.py

Modified into spec aug with masking by ratio by Huahuan Zheng (maxwellzh@outlook.com) in 2021.
"""

from .layer import StackDelta, UnStackDelta
from .coreutils import pad_list
from typing import *

import torch
import torch.nn as nn


class MaskFreq(nn.Module):
    def __init__(
        self,
        mask_width_range: Union[float, Sequence[float]] = (0., 0.15),
        num_mask: int = 2
    ):
        if isinstance(mask_width_range, float):
            mask_width_range = (0., mask_width_range)
        if len(mask_width_range) != 2:
            raise TypeError(
                f"mask_width_range must be a tuple of int and int values: "
                f"{mask_width_range}",
            )

        assert mask_width_range[1] > mask_width_range[0]
        assert mask_width_range[1] < 1. and mask_width_range[0] >= 0.

        super().__init__()
        self.mask_width_range = mask_width_range
        self.num_mask = num_mask

    def forward(self, spec: torch.Tensor):
        """Apply mask along the freq direction.
        Args:
            spec: (batch, length, freq) or (batch, channel, length, freq)
        """
        mask_width_range = self.mask_width_range
        idim = spec.size(-1)
        mask_width_range = (
            int(idim*mask_width_range[0]), int(idim*mask_width_range[1]))

        if mask_width_range[0] == mask_width_range[1]:
            return spec

        num_mask = self.num_mask
        org_size = spec.size()
        if spec.dim() == 4:
            # spec: (Batch, Channel, Length, Freq) -> (Batch * Channel, Length, Freq)
            spec = spec.view(-1, spec.size(2), spec.size(3))

        B = spec.shape[0]
        # D = Length or Freq
        D = spec.shape[-1]
        # mask_length: (B, num_mask, 1)
        mask_length = torch.randint(
            mask_width_range[0],
            mask_width_range[1],
            (B, num_mask),
            device=spec.device,
        ).unsqueeze(2)

        # mask_pos: (B, num_mask, 1)
        mask_pos = torch.randint(
            0, max(1, D - mask_length.max()), (B, num_mask), device=spec.device
        ).unsqueeze(2)

        # aran: (1, 1, D)
        aran = torch.arange(D, device=spec.device)[None, None, :]
        # mask: (Batch, num_mask, D)
        mask = (mask_pos <= aran) * (aran < (mask_pos + mask_length))
        # Multiply masks: (Batch, num_mask, D) -> (Batch, D)
        mask = mask.any(dim=1)

        # mask: (Batch, 1, Freq)
        mask = mask.unsqueeze(1)

        if spec.requires_grad:
            spec = spec.masked_fill(mask, 0.0)
        else:   # in-place fill
            spec = spec.masked_fill_(mask, 0.0)
        spec = spec.view(*org_size)
        return spec


class MaskTime(nn.Module):
    def __init__(
        self,
        mask_width_range: Union[float, Sequence[float]] = (0., 0.15),
        num_mask: int = 2
    ):
        if isinstance(mask_width_range, float):
            mask_width_range = (0., mask_width_range)
        if len(mask_width_range) != 2:
            raise TypeError(
                f"mask_width_range must be a tuple of int and int values: "
                f"{mask_width_range}",
            )

        assert mask_width_range[1] > mask_width_range[0]
        assert mask_width_range[1] < 1. and mask_width_range[0] >= 0.

        super().__init__()
        self.mask_width_range = mask_width_range
        self.num_mask = num_mask

    def mask_by_batch(self, x: torch.Tensor):
        org_size = x.size()
        if x.dim() == 4:
            # x: (Batch, Channel, Length, Freq) -> (Batch * Channel, Length, Freq)
            x = x.view(-1, x.size(2), x.size(3))

        mask_width_range = self.mask_width_range
        idim = x.size(1)
        mask_width_range = (
            int(idim*mask_width_range[0]), int(idim*mask_width_range[1]))

        if mask_width_range[0] == mask_width_range[1]:
            return x

        # the fused dimension of batch and channel
        fusedBC = x.size(0)
        L = x.size(1)
        mask_len = torch.randint(
            low=mask_width_range[0],
            high=mask_width_range[1],
            size=(fusedBC, self.num_mask),
            device=x.device
        ).unsqueeze(2)
        mask_pos = torch.randint(
            0, max(1, L - mask_len.max()), (fusedBC, self.num_mask), device=x.device
        ).unsqueeze(2)

        aran = torch.arange(L, device=x.device)[None, None, :]
        mask = (mask_pos <= aran) * (aran < (mask_pos + mask_len))
        mask = mask.any(dim=1).unsqueeze(2)

        if x.requires_grad:
            x = x.masked_fill(mask, 0.0)
        else:
            x = x.masked_fill_(mask, 0.0)

        x = x.view(*org_size)
        return x

    def forward(self, spec: torch.Tensor, spec_length: torch.Tensor):
        """Apply mask along time direction.
        Args:
            spec: (batch, length, freq) or (batch, channel, length, freq)
            spec_lengths: (length)
        """

        if all(le == spec_length[0] for le in spec_length):
            out = self.mask_by_batch(spec)
        else:
            org_size = spec.size()
            batch = spec.size(0)
            if spec.dim() == 4:
                ch = spec.size(1)
                # spec: (Batch, Channel, Length, Freq) -> (Batch*Channel, Length, Freq)
                spec = spec.view(-1, org_size[2], org_size[3])
            else:
                ch = 1
            outs = []
            for i in range(batch):
                for j in range(ch):
                    _out = self.mask_by_batch(
                        spec[i*ch+j][None, :spec_length[i], :])
                    outs.append(_out)
            out = pad_list(outs, 0.0, dim=1)
            out = out.view(*org_size)
        return out


def time_warp(x: torch.Tensor, window: int = 40, mode: str = "bicubic"):
    """Time warping using torch.interpolate.

    Args:
        x: (Batch, Time, Freq)
        window: time warp parameter
        mode: Interpolate mode
    """

    # bicubic supports 4D or more dimension tensor
    if window == 0:
        return x
    org_size = x.size()
    if x.dim() == 3:
        # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
        x = x[:, None]

    t = x.shape[2]
    if t - window <= window:
        return x.view(*org_size)

    center = torch.randint(window, t - window, (1,))[0]
    warped = torch.randint(center - window, center + window, (1,))[0] + 1

    # left: (Batch, Channel, warped, Freq)
    # right: (Batch, Channel, time - warped, Freq)
    left = torch.nn.functional.interpolate(
        x[:, :, :center], (warped, x.shape[3]), mode=mode, align_corners=False
    )
    right = torch.nn.functional.interpolate(
        x[:, :, center:], (t - warped, x.shape[3]), mode=mode, align_corners=False
    )

    if x.requires_grad:
        x = torch.cat([left, right], dim=-2)
    else:
        x[:, :, :warped] = left
        x[:, :, warped:] = right

    return x.view(*org_size)


class TimeWarp(torch.nn.Module):
    """Time warping using torch.interpolate.

    Args:
        window: time warp parameter
        mode: Interpolate mode
    """

    def __init__(self, window: float = 0.1):
        super().__init__()
        self.window = window

    def forward(self, spec: torch.Tensor, spec_lengths: torch.Tensor):
        """Forward function.

        Args:
            x: (Batch, Time, Freq) or (Batch, Channel, Time, Freq)
            x_lengths: (Batch,)
        """
        org_size = spec.size()
        batch = spec.size(0)
        if spec.dim() == 4:
            ch = spec.size(1)
            # spec: (Batch, Channel, Length, Freq) -> (Batch*Channel, Length, Freq)
            spec = spec.view(-1, org_size[2], org_size[3])
        else:
            ch = 1

        if all(le == spec_lengths[0] for le in spec_lengths):
            # Note that applying same warping for each sample
            y = time_warp(spec, window=int(spec_lengths[0]*self.window))
        else:
            ys = []
            for i in range(batch):
                _y = time_warp(spec[i*ch:i*ch+ch][:, :spec_lengths[i], :],
                               window=int(spec_lengths[i]*self.window))
                ys.append(_y)
            y = pad_list(ys, 0.0, dim=1)

        y = y.view(*org_size)

        return y


class SpecAug(nn.Module):
    """Implementation of SpecAug.

    Reference:
        Daniel S. Park et al.
        "SpecAugment: A Simple Data
         Augmentation Method for Automatic Speech Recognition"

    .. warning::
        When using cuda mode, time_warp doesn't have reproducibility
        due to `torch.nn.functional.interpolate`.

    """

    def __init__(
        self,
        apply_time_warp: bool = True,
        time_warp_window: float = 0.2,
        apply_freq_mask: bool = True,
        freq_mask_width_range: Union[float, Sequence[float]] = (0., 0.15),
        num_freq_mask: int = 2,
        apply_time_mask: bool = True,
        time_mask_width_range: Union[float, Sequence[float]] = (0., 0.1),
        num_time_mask: int = 2,
        delta_feats: bool = False
    ):
        if not apply_time_warp and not apply_time_mask and not apply_freq_mask:
            raise ValueError(
                "Either one of time_warp, time_mask, or freq_mask should be applied",
            )
        super().__init__()
        self._lens_in_args_ = None
        self.delta_ = delta_feats

        if delta_feats:
            self.stack = StackDelta()
            self.unstack = UnStackDelta()

        if apply_time_warp and time_warp_window > 0.0:
            self.time_warp = TimeWarp(
                window=time_warp_window)
        else:
            self.time_warp = None

        if apply_freq_mask:
            self.freq_mask = MaskFreq(
                mask_width_range=freq_mask_width_range,
                num_mask=num_freq_mask,
            )
        else:
            self.freq_mask = None

        if apply_time_mask:
            self.time_mask = MaskTime(
                mask_width_range=time_mask_width_range,
                num_mask=num_time_mask,
            )
        else:
            self.time_mask = None

    def forward(self, x, x_lengths=None):
        if not self.training:
            return x, x_lengths

        if self.time_warp is not None:
            x = self.time_warp(x, x_lengths)

        if self.time_mask is not None:
            x = self.time_mask(x, x_lengths)

        if self.delta_:
            x = self.stack(x)

        if self.freq_mask is not None:
            x = self.freq_mask(x)

        if self.delta_:
            x = self.unstack(x)
        return x, x_lengths
