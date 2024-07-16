# Copyright 2020 Tsinghua SPMI Lab 
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com), Keyu An
#
# Acknowledgment:
#   This code is adapted from the ESPnet project. The original code can be found at https://github.com/espnet/espnet.
#
# Description:
#   This script defines the FilterNet class, which is designed to apply filtering 
#   to multi-channel complex spectrograms using a neural network. It includes 
#   functionality for processing complex input signals, computing masks, and 
#   generating filtered outputs.

import torch
import numpy as np
from .mask_estimator import RNNP, RNN
from torch_complex.tensor import ComplexTensor
from typing import Tuple
from torch.nn import functional as F

class FilterNet(torch.nn.Module):
    """
    A neural network module for filtering complex tensors using RNN layers.

    Args:
        type (str): Type of RNN to use (e.g., 'lstm', 'gru', etc.).
        idim (int): Input dimension.
        layers (int): Number of RNN layers.
        units (int): Number of units in each RNN layer.
        projs (int): Number of projection units.
        dropout (float): Dropout rate.
        nmask (int, optional): Number of masks. Default is 1.

    Methods:
        forward(xs: ComplexTensor, ilens: torch.LongTensor) -> Tuple[Tuple[torch.Tensor, ...], torch.LongTensor]:
            Forward propagation through the network.
    """
    def __init__(self, type, idim, layers, units, projs, dropout, nmask=1):
        super().__init__()
        subsample = np.ones(layers + 1, dtype=np.int)

        typ = type.lstrip("vgg").rstrip("p")
        if type[-1] == "p":
            self.brnn = RNNP(idim, layers, units, projs, subsample, dropout, typ=typ)
        else:
            self.brnn = RNN(idim, layers, units, projs, dropout, typ=typ)

        self.type = type
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(projs, idim) for _ in range(2)]
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
        #print(xs.shape)
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

        xs = xs.view(-1, C, xs.size(-2), xs.size(-1))
        fltrs = []
        for linear in self.linears:
        # xs: (B, C, T, D) -> mask:(B, C, T, F)
            fltr = linear(xs)
            fltr = torch.tanh(fltr)
            fltr = fltr.permute(0, 3, 1, 2)
            
            if fltr.size(-1) < input_length:
                fltr = F.pad(fltr, [0, input_length - fltr.size(-1)], value=0)
            fltrs.append(fltr)
        fltrs = ComplexTensor(fltrs[0],fltrs[1])
        return fltrs
        #print(fltrs.conj().shape)
        #print(xs.shape)
        #es = FC.einsum("...ct,...ct->...t", [fltrs.conj(), xs])
        #print(es.shape)
        #exit()


