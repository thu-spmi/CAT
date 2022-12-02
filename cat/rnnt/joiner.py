# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""Joint network modules.
"""


__all__ = ["AbsJointNet", "JointNet"]


import gather
from typing import *

import torch
import torch.nn as nn


class AbsJointNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def is_normalize_separated(self) -> bool:
        """ Tell if the log_softmax could be split from forward function,
            useful for Transducer fused rnnt loss
        """
        return True

    def impl_forward(self, *args, **kwargs):
        """forward without log_softmax"""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        joinout = self.impl_forward(
            *args, **kwargs)  # type: torch.Tensor
        return joinout.log_softmax(dim=-1)


class JointNet(AbsJointNet):
    """
    Joint `encoder_output` and `predictor_output`.
    Args:
        encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size ``(batch, time_steps, dimensionA)``
        predictor_output (torch.FloatTensor): A output sequence of predictor. `FloatTensor` of size ``(batch, label_length, dimensionB)``
    Returns:
        outputs (torch.FloatTensor): outputs of joined `encoder_output` and `predictor_output`. `FloatTensor` of size ``(batch, time_steps, label_length, dimensionA + dimensionB)``
    """

    def __init__(
            self,
            odim_enc: int,
            odim_pred: int,
            num_classes: int,
            hdim: int = -1,
            join_mode: Literal['add', 'cat'] = 'add',
            act: Literal['tanh', 'relu'] = 'tanh',
            compact: bool = False,
            pre_project: bool = True):
        super().__init__()

        if join_mode == 'add':
            if hdim == -1:
                hdim = max(odim_pred, odim_enc)

            if pre_project:
                self.fc_enc = nn.Linear(odim_enc, hdim)
                self.fc_dec = nn.Linear(odim_pred, hdim)
            else:
                assert odim_enc == odim_pred
                self.fc_enc = nn.Identity()
                self.fc_dec = nn.Identity()
        elif join_mode == 'cat':
            self.fc_enc = None
            self.fc_dec = None
            hdim = odim_enc+odim_pred
        else:
            raise RuntimeError(f"Unknown mode for joint net: {join_mode}")

        if act == 'tanh':
            act_layer = nn.Tanh()
        elif act == 'relu':
            act_layer = nn.ReLU()
        else:
            raise NotImplementedError(f"Unknown activation layer type: {act}")
        self.fc = nn.Sequential(
            act_layer,
            nn.Linear(hdim, num_classes)
        )
        self._mode = join_mode
        self.iscompact = compact

    def impl_forward(self, enc_out: torch.Tensor, pred_out: torch.Tensor, enc_out_lens: Optional[torch.IntTensor] = None, pred_out_lens: Optional[torch.IntTensor] = None) -> torch.FloatTensor:

        d_enc = enc_out.dim()
        assert d_enc == pred_out.dim(), \
            f"expect encoder output and decoder output to be the same dimentional, " \
            f"instead {d_enc} != {pred_out.dim()}"
        assert d_enc == 1 or d_enc == 3, f"only support input dimension is 1 or 3, instead {d_enc}"

        if d_enc == 3 and self.iscompact and enc_out_lens is not None and pred_out_lens is not None:
            enc_out = gather.cat(enc_out, enc_out_lens)
            pred_out = gather.cat(pred_out, pred_out_lens)
            d_enc = 2

        if self._mode == 'add':
            enc_out = self.fc_enc(enc_out)
            pred_out = self.fc_dec(pred_out)
            if d_enc == 1:
                # streaming inference mode
                expanded_out = enc_out + pred_out
            elif d_enc == 2:
                # compact layout
                expanded_out = gather.sum(
                    enc_out, pred_out, enc_out_lens, pred_out_lens)
            else:  # d_enc == 3
                # normal layout, use broadcast sum
                expanded_out = enc_out[:, :, None, :] + pred_out[:, None, :, :]
        elif self._mode == 'cat':
            if d_enc == 1:
                expanded_out = torch.cat([enc_out, pred_out], dim=-1)
            elif d_enc == 2:
                # this is not efficient, so better avoid using 'cat' mode with compact layout
                v_enc, v_pred = enc_out.size(-1), pred_out.size(-1)
                enc_out = torch.nn.functional.pad(enc_out, (0, v_pred))
                pred_out = torch.nn.functional.pad(pred_out, (v_enc, 0))
                expanded_out = gather.sum(
                    enc_out, pred_out, enc_out_lens, pred_out_lens)
            else:  # d_enc == 3
                T, Up = enc_out.size(1), pred_out.size(1)
                enc_out = enc_out[:, :, None, :].expand(-1, -1, Up, -1)
                pred_out = pred_out[:, None, :, :].expand(-1, T, -1, -1)
                expanded_out = torch.cat([enc_out, pred_out], dim=-1)
        else:
            raise ValueError(
                f"Unknown joint mode: {self._mode}, expect one of ['add', 'cat']")

        return self.fc(expanded_out)

    def forward_pred_only(self, pred_out: torch.Tensor, raw_logit: bool = False):
        if self._mode == 'add':
            cast_pred = self.fc(self.fc_dec(pred_out))
        elif self._mode == 'cat':
            pred_out = torch.nn.functional.pad(
                pred_out, (0, self.fc[1].in_features - pred_out.size(-1)))
            cast_pred = self.fc(pred_out)
        else:
            raise ValueError(self._mode)

        if raw_logit:
            return cast_pred
        else:
            return cast_pred.log_softmax(dim=-1)


class HAT(JointNet):
    """ "HYBRID AUTOREGRESSIVE TRANSDUCER (HAT)"

    Suppose <blk>=0
    """

    def __init__(
            self,
            odim_enc: int,
            odim_pred: int,
            num_classes: int,
            hdim: int = -1,
            join_mode: Literal['add', 'cat'] = 'add',
            act: Literal['tanh', 'relu'] = 'tanh'):
        super().__init__(odim_enc, odim_pred, num_classes,
                         hdim=hdim, join_mode=join_mode, act=act)
        self._dist_blank = nn.LogSigmoid()

    @property
    def is_normalize_separated(self) -> bool:
        return False

    def ilm_est(self, pred_out: torch.Tensor, pred_out_lens: Optional[torch.IntTensor]):
        """ILM score estimation"""
        assert not self.training

        fc_out = self.fc(self.fc_dec(pred_out))
        # suppose blank=0
        # compute log softmax over real labels
        fc_out[..., 1:] = fc_out[..., 1:].log_softmax(dim=-1)
        return fc_out

    def forward(self, *args, **kwargs):
        # [..., V]
        logits = super().impl_forward(*args, **kwargs)
        # [..., 1]
        logit_blank = logits[..., 0:1]
        log_prob_blank = self._dist_blank(logit_blank)
        # FIXME: maybe we should cast it to float for numerical stablility
        # sigmoid(x) = 1/(1+exp(-x)) ->
        # 1-sigmoid(x) = 1/(1+exp(x)) = sigmoid(-x)
        # [..., V-1]
        log_prob_label = logits[..., 1:].log_softmax(
            dim=-1) + self._dist_blank(-logit_blank)
        return torch.cat([log_prob_blank, log_prob_label], dim=-1)


class LogAdd(AbsJointNet):
    def __init__(self, compact: bool = False) -> None:
        super().__init__()
        self.iscompact = compact

    def impl_forward(self, f: torch.Tensor, g: torch.Tensor, lf: torch.Tensor = None, lg: torch.Tensor = None):
        assert f.dim() == g.dim()
        dim_f = f.dim()
        assert dim_f == 1 or dim_f == 3, f"only support input dimension is 1 or 3, instead {dim_f}"

        # the preditor doesn't straightly involve in blank prob computation.
        # g[..., 0] = 0.
        if dim_f == 3 and self.iscompact and lf is not None and lg is not None:
            return torch.cat([
                (f[i:i+1, :lf[i]].unsqueeze(2) +
                 g[i:i+1, :lg[i]].unsqueeze(1)).view(-1, f.size(-1))
                for i in range(f.size(0))
            ], dim=0)
        if dim_f == 1:
            # streaming inference mode
            return f + g
        else:
            # normal padding mode
            return f.unsqueeze(2) + g.unsqueeze(1)

