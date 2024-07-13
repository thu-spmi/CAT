# Copyright 2020 Tsinghua SPMI Lab / Tasi
# Apache 2.0.
# Author: Xiangzhu (kongxiangzhu99@gmail.com), Keyu An
#
# Acknowledgment:
# This code is adapted from the ESPnet project. The original code can be found at https://github.com/espnet/espnet.
#
# Description:
#   This script defines several classes and functions for processing and estimating masks for speech enhancement using RNNs.

import torch
import numpy as np
from torch_complex.tensor import ComplexTensor
from typing import Tuple
import six
import logging
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from .nets_utils import make_pad_mask


def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes

    Useful in processing of sliding windows over the inputs
    """
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.0
    else:
        states[1::2] = 0.0
    return states


class RNNP(torch.nn.Module):
    """RNN with projection layer module
    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of projection units
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout, typ="blstm"):
        super(RNNP, self).__init__()
        bidir = typ[0] == "b"
        for i in six.moves.range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim

            RNN = torch.nn.LSTM if "lstm" in typ else torch.nn.GRU
            rnn = RNN(
                inputdim, cdim, num_layers=1, bidirectional=bidir, batch_first=True
            )

            setattr(self, "%s%d" % ("birnn" if bidir else "rnn", i), rnn)

            # bottleneck layer to merge
            if bidir:
                setattr(self, "bt%d" % i, torch.nn.Linear(2 * cdim, hdim))
            else:
                setattr(self, "bt%d" % i, torch.nn.Linear(cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample
        self.typ = typ
        self.bidir = bidir
        self.dropout = dropout

    def forward(self, xs_pad, ilens, prev_state=None):
        """RNNP forward
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, hdim)
        :rtype: torch.Tensor
        """
        logging.debug(self.__class__.__name__ + " input lengths: " + str(ilens))
        elayer_states = []
        for layer in six.moves.range(self.elayers):
            xs_pack = pack_padded_sequence(xs_pad, ilens.cpu().int(), batch_first=True)
            rnn = getattr(self, ("birnn" if self.bidir else "rnn") + str(layer))
            rnn.flatten_parameters()
            if prev_state is not None and rnn.bidirectional:
                prev_state = reset_backward_rnn_state(prev_state)
            ys, states = rnn(
                xs_pack, hx=None if prev_state is None else prev_state[layer]
            )
            elayer_states.append(states)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ys_pad = ys_pad[:, ::sub]
                ilens = [int(i + 1) // sub for i in ilens]
            # (sum _utt frame_utt) x dim
            projection_layer = getattr(self, "bt%d" % layer)
            projected = projection_layer(ys_pad.contiguous().view(-1, ys_pad.size(2)))
            xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
            if layer < self.elayers - 1:
                xs_pad = torch.tanh(F.dropout(xs_pad, p=self.dropout))

        return xs_pad, ilens, elayer_states  # x: utt list of frame x dim

class RNN(torch.nn.Module):
    """RNN module
    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of final projection units
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout, typ="blstm"):
        super(RNN, self).__init__()
        bidir = typ[0] == "b"
        self.nbrnn = (
            torch.nn.LSTM(
                idim,
                cdim,
                elayers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidir,
            )
            if "lstm" in typ
            else torch.nn.GRU(
                idim,
                cdim,
                elayers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidir,
            )
        )
        if bidir:
            self.l_last = torch.nn.Linear(cdim * 2, hdim)
        else:
            self.l_last = torch.nn.Linear(cdim, hdim)
        self.typ = typ

    def forward(self, xs_pad, ilens, prev_state=None):
        """RNN forward
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        logging.debug(self.__class__.__name__ + " input lengths: " + str(ilens))
        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        self.nbrnn.flatten_parameters()
        if prev_state is not None and self.nbrnn.bidirectional:
            # We assume that when previous state is passed,
            # it means that we're streaming the input
            # and therefore cannot propagate backward BRNN state
            # (otherwise it goes in the wrong direction)
            prev_state = reset_backward_rnn_state(prev_state)
        ys, states = self.nbrnn(xs_pack, hx=prev_state)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(
            self.l_last(ys_pad.contiguous().view(-1, ys_pad.size(2)))
        )
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
        return xs_pad, ilens, states  # x: utt list of frame x dim

class MaskEstimator(torch.nn.Module):
    def __init__(self, type, idim, layers, units, projs, dropout, nmask=1):
        super().__init__()
        subsample = np.ones(layers + 1, dtype=np.int)

        typ = type.lstrip("vgg").rstrip("p")
        if type[-1] == "p":
            self.brnn = RNNP(idim, layers, units, projs, subsample, dropout, typ=typ)
        else:
            self.brnn = RNN(idim, layers, units, projs, dropout, typ=typ)

        self.type = type
        self.nmask = nmask
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(projs, idim) for _ in range(nmask)]
        )

    def forward(
        self, xs: ComplexTensor, ilens: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.LongTensor]:
        """The forward function
        Args:
            xs: (B, F, C, T)
            ilens: (B,)
        Returns:
            hs (torch.Tensor): The hidden vector (B, F, C, T)
            masks: A tuple of the masks. (B, F, C, T)
            ilens: (B,)
        """
        assert xs.size(0) == ilens.size(0), (xs.size(0), ilens.size(0))
        _, _, C, input_length = xs.size()
        # (B, F, C, T) -> (B, C, T, F)
        xs = xs.permute(0, 2, 3, 1)

        # Calculate amplitude: (B, C, T, F) -> (B, C, T, F)
        xs = (xs.real ** 2 + xs.imag ** 2) ** 0.5
        # xs: (B, C, T, F) -> xs: (B * C, T, F)
        xs = xs.contiguous().view(-1, xs.size(-2), xs.size(-1))
        # ilens: (B,) -> ilens_: (B * C)
        ilens_ = ilens[:, None].expand(-1, C).contiguous().view(-1)

        # xs: (B * C, T, F) -> xs: (B * C, T, D)
        xs, _, _ = self.brnn(xs, ilens_)
        # xs: (B * C, T, D) -> xs: (B, C, T, D)
        xs = xs.view(-1, C, xs.size(-2), xs.size(-1))

        masks = []
        for linear in self.linears:
            # xs: (B, C, T, D) -> mask:(B, C, T, F)
            mask = linear(xs)

            mask = torch.sigmoid(mask)
            # Zero padding
            mask.masked_fill(make_pad_mask(ilens, mask, length_dim=2), 0)

            # (B, C, T, F) -> (B, F, C, T)
            mask = mask.permute(0, 3, 1, 2)

            # Take cares of multi gpu cases: If input_length > max(ilens)
            if mask.size(-1) < input_length:
                mask = F.pad(mask, [0, input_length - mask.size(-1)], value=0)
            masks.append(mask)

        return tuple(masks), ilens
