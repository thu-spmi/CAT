"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Keyu An, Huahuan Zheng

In this file, we define universal models 
"""

import numpy as np
import _layers as nnlayers
from collections import OrderedDict

import torch
import torch.nn as nn


def get_vgg2l_odim(idim, in_channel=1, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels


class LSTM(nn.Module):
    def __init__(self, idim, hdim, n_layers, num_classes, dropout, bidirectional=False):
        super().__init__()
        self.lstm = nnlayers._LSTM(
            idim, hdim, n_layers, dropout, bidirectional=bidirectional)

        if bidirectional:
            self.linear = nn.Linear(hdim * 2, num_classes)
        else:
            self.linear = nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor, hidden=None):
        lstm_out, olens = self.lstm(x, ilens, hidden)
        out = self.linear(lstm_out)
        return out, olens


class BLSTM(LSTM):
    def __init__(self, idim, hdim, n_layers, num_classes, dropout):
        super().__init__(idim, hdim, n_layers, num_classes, dropout, bidirectional=True)


class VGGLSTM(LSTM):
    def __init__(self, idim, hdim, n_layers, num_classes, dropout, in_channel=3, bidirectional=False):
        super().__init__(get_vgg2l_odim(idim, in_channel=in_channel), hdim,
                         n_layers, num_classes, dropout, bidirectional=bidirectional)

        self.VGG = nnlayers.VGG2L(in_channel)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor):
        vgg_o, vgg_lens = self.VGG(x, ilens)
        return super().forward(vgg_o, vgg_lens)


class VGGBLSTM(VGGLSTM):
    def __init__(self, idim, hdim, n_layers, num_classes, dropout, in_channel=3):
        super().__init__(idim, hdim, n_layers, num_classes,
                         dropout, in_channel=in_channel, bidirectional=True)


class LSTMrowCONV(nn.Module):
    def __init__(self, idim, hdim, n_layers, num_classes, dropout):
        super().__init__()

        self.lstm = nnlayers._LSTM(idim, hdim, n_layers, dropout)
        self.lookahead = nnlayers.Lookahead(hdim, context=5)
        self.linear = nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor, hidden=None):
        lstm_out, olens = self.lstm(x, ilens, hidden)
        ahead_out = self.lookahead(lstm_out)
        return self.linear(ahead_out), olens


class TDNN_NAS(torch.nn.Module):
    def __init__(self, idim: int, hdim: int,  num_classes: int, dropout: float = 0.5):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.tdnns = nn.ModuleDict(OrderedDict({
            'tdnn0': nnlayers.TDNN(idim, hdim, half_context=2, dilation=1),
            'tdnn1': nnlayers.TDNN(idim, hdim, half_context=2, dilation=2),
            'tdnn2': nnlayers.TDNN(idim, hdim, half_context=2, dilation=1),
            'tdnn3': nnlayers.TDNN(idim, hdim, stride=3),
            'tdnn4': nnlayers.TDNN(idim, hdim, half_context=2, dilation=2),
            'tdnn5': nnlayers.TDNN(idim, hdim, half_context=2, dilation=1),
            'tdnn6': nnlayers.TDNN(idim, hdim, half_context=2, dilation=2)
        }))

        self.linear = nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor):
        tmp_x, tmp_lens = x, ilens
        for i, tdnn in enumerate(self.tdnns.values):
            if i < len(self.tdnns)-1:
                tmp_x = self.dropout(x)
            tmp_x, tmp_lens = tdnn(tmp_x, tmp_lens)

        return self.linear(tmp_x), tmp_lens


class TDNN_LSTM(torch.nn.Module):
    def __init__(self, idim: int, hdim: int, n_layers: int, num_classes: int,  dropout: float):
        super().__init__()

        self.tdnn_init = nnlayers.TDNN(idim, hdim)
        assert n_layers > 0
        self.n_layers = n_layers
        self.cells = nn.ModuleDict()
        for i in range(n_layers):
            self.cells[f"tdnn{i}-0"] = nnlayers.TDNN(hdim, hdim)
            self.cells[f"tdnn{i}-1"] = nnlayers.TDNN(hdim, hdim)
            self.cells[f"lstm{i}"] = nnlayers._LSTM(hdim, hdim, 1)
            self.cells[f"bn{i}"] = nnlayers.MaskedBatchNorm1d(
                hdim, eps=1e-5, affine=True)
            self.cells[f"dropout{i}"] = nn.Dropout(dropout)

        self.linear = nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor):

        tmp_x, tmp_lens = self.tdnn_init(x, ilens)

        for i in range(self.n_layers):
            tmpx, tmp_lens = self.cells[f"tdnn{i}-0"](tmpx, tmp_lens)
            tmpx, tmp_lens = self.cells[f"tdnn{i}-1"](tmpx, tmp_lens)
            tmpx, tmp_lens = self.cells[f"lstm{i}"](tmpx, tmp_lens)
            tmpx = self.cells[f"bn{i}"](tmpx, tmp_lens)
            tmpx = self.cells[f"dropout{i}"](tmpx)

        return self.linear(tmpx), tmp_lens


class BLSTMN(torch.nn.Module):
    def __init__(self, idim: int, hdim: int, n_layers: int, num_classes: int,  dropout: float):
        super(BLSTMN, self).__init__()
        assert n_layers > 0
        self.cells = nn.ModuleDict()
        self.n_layers = n_layers
        for i in range(n_layers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim * 2
            self.cells[f"lstm{i}"] = nnlayers._LSTM(
                inputdim, hdim, 1, bidirectional=True)
            self.cells[f"bn{i}"] = nnlayers.MaskedBatchNorm1d(
                hdim*2, eps=1e-5, affine=True)
            self.cells[f"dropout{i}"] = nn.Dropout(dropout)

        self.linear = nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, ilens: torch.Tensor):
        tmp_x, tmp_lens = x, ilens
        for i in range(self.n_layers):
            tmpx, tmp_lens = self.cells[f"lstm{i}"](tmpx, tmp_lens)
            tmpx = self.cells[f"bn{i}"](tmpx, tmp_lens)
            tmpx = self.cells[f"dropout{i}"](tmpx)

        return self.linear(tmpx), tmp_lens


class ConformerNet(nn.Module):
    """The conformer model with convolution subsampling

    Args:
        num_cells (int): number of conformer blocks
        idim (int): dimension of input features
        hdim (int): hidden size in conformer blocks
        num_classes (int): number of output classes
        conv_multiplier (int): the multiplier to conv subsampling module
        dropout_in (float): the dropout rate to input of conformer blocks (after the linear and subsampling layers)
        res_factor (float): the weighted-factor of residual-connected shortcut in feed-forward module
        d_head (int): dimension of heads in multi-head attention module
        num_heads (int): number of heads in multi-head attention module
        kernel_size (int): kernel size in convolution module
        multiplier (int): multiplier of depth conv in convolution module 
        dropout (float): dropout rate to all conformer internal modules
        delta_feats (bool): True if the input features contains delta and delta-delta features; False if not.
    """

    def __init__(
            self,
            num_cells: int,
            idim: int,
            hdim: int,
            num_classes: int,
            conv_multiplier: int = 144,
            dropout_in: float = 0.2,
            res_factor: float = 0.5,
            d_head: int = 36,
            num_heads: int = 4,
            kernel_size: int = 32,
            multiplier: int = 1,
            dropout: float = 0.1,
            delta_feats=False):
        super().__init__()

        if delta_feats:
            idim = idim // 3
        self.conv_subsampling = nnlayers.Conv2dSubdampling(
            conv_multiplier, stacksup=delta_feats)
        self.linear_drop = nn.Sequential(OrderedDict({
            'linear': nn.Linear((idim // 4) * conv_multiplier, hdim),
            'dropout': nn.Dropout(dropout_in)
        }))
        self.cells = nn.ModuleList()
        pe = nnlayers.PositionalEncoding(hdim)
        for i in range(num_cells):
            cell = nnlayers.ConformerCell(
                hdim, res_factor, d_head, num_heads, kernel_size, multiplier, dropout)
            self.cells.append(cell)
            # FIXME: Note that this is somewhat hard-code style
            cell.mhsam.mha.pe = pe
        self.classifier = nn.Linear(hdim, num_classes)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        x_subsampled, ls_subsampled = self.conv_subsampling(x, lens)
        out = self.linear_drop(x_subsampled)
        ls = ls_subsampled
        for cell in self.cells:
            out, ls = cell(out, ls)
        logits = self.classifier(out)

        return logits, ls


# TODO: (Huahuan) I removed all chunk-related modules.
#       cc @aky15 you may need to add it in v2 standard
