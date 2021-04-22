"""
Copyright 2020 Tsinghua University
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
In this file, we implement the basic conformer 
"""

from . import _layers as nn_layers
from collections import OrderedDict

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFModule(nn.Module):
    """Feed-forward module

    default output dimension = idim
    x0 -> LayerNorm -> FC -> Swish -> Dropout -> FC -> Dropout -> x1
    x0 + res_factor * x1 -> output
    """

    def __init__(self, idim: int, res_factor: float = 0.5, dropout: float = 0.0) -> None:
        super().__init__()
        assert res_factor > 0. and dropout >= 0.
        self._res_factor = res_factor

        self.ln = nn.LayerNorm([idim])
        self.fc0 = nn.Linear(idim, idim*4)
        self.swish = nn.SiLU()
        self.dropout0 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(idim*4, idim)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        output = self.ln(x)
        output = self.fc0(output)
        output = self.swish(output)
        output = self.dropout0(output)
        output = self.fc1(output)
        output = self.dropout1(output)
        output = x + self._res_factor * output

        return output


class ConvModule(nn.Module):
    def __init__(self, idim: int, kernel_size: int = 32, dropout: float = 0.0, multiplier: int = 1) -> None:
        super().__init__()

        self.ln = nn.LayerNorm([idim])
        self.pointwise_conv0 = nn.Conv1d(
            idim, 2 * idim, kernel_size=1, stride=1)
        self.glu = nn.GLU(dim=1)
        cdim = idim
        padding = (kernel_size-1)//2
        self.padding = nn.ConstantPad2d((padding, kernel_size-1-padding, 0, 0), 0.)
        self.depthwise_conv = nn.Conv1d(
            cdim, multiplier*cdim, kernel_size=kernel_size, stride=1, groups=cdim, padding=0)
        cdim = multiplier * cdim
        self.bn = nn.BatchNorm1d(cdim)
        self.swish = nn.SiLU()
        self.pointwise_conv1 = nn.Conv1d(cdim, idim, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # [B, T, D]
        output = self.ln(x)
        # [B, T, D] -> [B, D, T]
        output = output.transpose(1, 2)
        # [B, D, T] -> [B, 2*D, T]
        output = self.pointwise_conv0(output)
        # [B, 2D, T] -> [B, D, T]
        output = self.glu(output)
        # [B, D, T] -> [B, multiplier*D, T]
        output = self.padding(output)
        output = self.depthwise_conv(output)
        if output.size(0) > 1 or output.size(2) > 1:
            # Doing batchnorm with [1, D, 1] tensor raise error.
            output = self.bn(output)
        output = self.swish(output)
        # [B, multiplier*D, T] -> [B, D, T]
        output = self.pointwise_conv1(output)
        output = self.dropout(output)
        # [B, D, T] -> [B, T, D]
        output = output.transpose(1, 2)

        return x + output


class MHSAModule(nn.Module):
    def __init__(self, idim, d_head: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        self.ln = nn.LayerNorm(idim)
        self.mha = nn_layers.RelPositionMultiHeadAttention(
            idim, num_heads, d_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lens: torch.Tensor, mems=None):
        x_norm = self.ln(x)
        attn_out = self.mha(x_norm, lens, mems)
        attn_out = self.dropout(attn_out)
        return x + attn_out, lens


class ConformerCell(nn.Module):
    def __init__(
            self,
            idim: int,
            res_factor: float = 0.5,
            d_head: int = 36,
            num_heads: int = 4,
            kernel_size: int = 32,
            multiplier: int = 1,
            dropout: float = 0.1):
        super().__init__()

        self.ffm0 = FFModule(idim, res_factor, dropout)
        self.mhsam = MHSAModule(idim, d_head, num_heads, dropout)
        self.convm = ConvModule(idim, kernel_size, dropout, multiplier)
        self.ffm1 = FFModule(idim, res_factor, dropout)
        self.ln = nn.LayerNorm(idim)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):

        ffm0_out = self.ffm0(x)
        attn_out, attn_ls = self.mhsam(ffm0_out, lens)
        conv_out = self.convm(attn_out)
        ffm1_out = self.ffm1(conv_out)
        out = self.ln(ffm1_out)
        return out, attn_ls


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
            delta_feats=True):

        if delta_feats:
            idim = idim // 3
        self.conv_subsampling = nn_layers.Conv2dSubdampling(
            conv_multiplier, stacksup=delta_feats)
        self.linear_drop = nn.Sequential(OrderedDict({
            'linear': nn.Linear((idim // 4) * conv_multiplier, hdim),
            'dropout': nn.Dropout(dropout_in)
        }))
        self.cells = nn.ModuleList()
        pe = nn_layers.PositionalEncoding(hdim)
        for i in range(num_cells):
            cell = ConformerCell(
                hdim, res_factor, d_head, num_heads, kernel_size, multiplier, dropout)
            self.cells.append(cell)
            # Note that this is somewhat hard-code style
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
