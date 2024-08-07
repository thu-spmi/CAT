# Copyright 2020 Tsinghua SPMI Lab 
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com), Keyu An
#
# Acknowledgment:
#   This code is adapted from the ESPnet project. The original code can be found at https://github.com/espnet/espnet.
#
# Description:
#   This script defines the DNN_WPE class, which implements a deep neural network 
#   based Weighted Prediction Error (WPE) method for dereverberation of multi-channel 
#   speech signals. The class includes functionality for estimating masks using a 
#   neural network and performing WPE iterations to enhance speech signals.

from typing import Tuple

from pytorch_wpe import wpe_one_iteration
import torch
from torch_complex.tensor import ComplexTensor

from .nets_utils import make_pad_mask
from .mask_estimator import MaskEstimator


class DNN_WPE(torch.nn.Module):
    """
    Implements a deep neural network based Weighted Prediction Error (WPE) method
    for dereverberation of multi-channel speech signals.

    Args:
        wtype (str): Type of neural network for mask estimation (default: "blstmp").
        widim (int): Input dimension of the neural network (default: 257).
        wlayers (int): Number of layers in the neural network (default: 3).
        wunits (int): Number of units in each layer of the neural network (default: 300).
        wprojs (int): Number of projection units in each layer of the neural network (default: 320).
        dropout_rate (float): Dropout rate for the neural network (default: 0.0).
        taps (int): Number of taps for WPE (default: 5).
        delay (int): Delay for WPE (default: 3).
        use_dnn_mask (bool): Whether to use DNN-based mask estimation (default: True).
        nmask (int): Number of masks (default: 1).
        nonlinear (str): Nonlinear activation function for mask estimation (default: "sigmoid").
        iterations (int): Number of WPE iterations (default: 1).
        normalization (bool): Whether to normalize masks (default: False).
        eps (float): Small value to avoid division by zero (default: 1e-6).
        diagonal_loading (bool): Whether to use diagonal loading for WPE (default: True).
        diag_eps (float): Small value for diagonal loading (default: 1e-7).
        mask_flooring (bool): Whether to floor masks for numerical stability (default: False).
        flooring_thres (float): Flooring threshold for masks (default: 1e-6).
        use_torch_solver (bool): Whether to use torch solver for WPE (default: True).

    Methods:
        forward(data: ComplexTensor, ilens: torch.LongTensor) -> Tuple[ComplexTensor, torch.LongTensor, ComplexTensor]:
            Forward function for WPE dereverberation.

        predict_mask(data: ComplexTensor, ilens: torch.LongTensor) -> Tuple[torch.Tensor, torch.LongTensor]:
            Predicts masks for WPE dereverberation.
    """
    def __init__(
        self,
        wtype: str = "blstmp",
        widim: int = 257,
        wlayers: int = 3,
        wunits: int = 300,
        wprojs: int = 320,
        dropout_rate: float = 0.0,
        taps: int = 5,
        delay: int = 3,
        use_dnn_mask: bool = True,
        nmask: int = 1,
        nonlinear: str = "sigmoid",
        iterations: int = 1,
        normalization: bool = False,
        eps: float = 1e-6,
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        mask_flooring: bool = False,
        flooring_thres: float = 1e-6,
        use_torch_solver: bool = True,
    ):
        super().__init__()
        self.iterations = iterations
        self.taps = taps
        self.delay = delay
        self.eps = eps

        self.normalization = normalization
        self.use_dnn_mask = use_dnn_mask

        self.inverse_power = True
        self.diagonal_loading = diagonal_loading
        self.diag_eps = diag_eps
        self.mask_flooring = mask_flooring
        self.flooring_thres = flooring_thres
        self.use_torch_solver = use_torch_solver

        if self.use_dnn_mask:
            self.nmask = nmask
            self.mask_est = MaskEstimator(
                wtype,
                widim,
                wlayers,
                wunits,
                wprojs,
                dropout_rate,
                nmask=nmask,
                nonlinear=nonlinear,
            )
        else:
            self.nmask = 1

    def forward(
        self, data: ComplexTensor, ilens: torch.LongTensor
    ) -> Tuple[ComplexTensor, torch.LongTensor, ComplexTensor]:
        """DNN_WPE forward function.
        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq or Some dimension of the feature vector
        Args:
            data: (B, T, C, F)
            ilens: (B,)
        Returns:
            enhanced (torch.Tensor or List[torch.Tensor]): (B, T, C, F)
            ilens: (B,)
            masks (torch.Tensor or List[torch.Tensor]): (B, T, C, F)
            power (List[torch.Tensor]): (B, F, T)
        """
        # (B, T, C, F) -> (B, F, C, T)
        data = data.permute(0, 3, 2, 1)
        enhanced = [data for i in range(self.nmask)]
        masks = None
        power = None

        for i in range(self.iterations):
            # Calculate power: (..., C, T)
            power = [enh.real ** 2 + enh.imag ** 2 for enh in enhanced]
            if i == 0 and self.use_dnn_mask:
                # mask: (B, F, C, T)
                masks, _ = self.mask_est(data, ilens)
                # floor masks to increase numerical stability
                if self.mask_flooring:
                    masks = [m.clamp(min=self.flooring_thres) for m in masks]
                if self.normalization:
                    # Normalize along T
                    masks = [m / m.sum(dim=-1, keepdim=True) for m in masks]
                # (..., C, T) * (..., C, T) -> (..., C, T)
                power = [p * masks[i] for i, p in enumerate(power)]

            # Averaging along the channel axis: (..., C, T) -> (..., T)
            power = [p.mean(dim=-2).clamp(min=self.eps) for p in power]

            # enhanced: (..., C, T) -> (..., C, T)
            # NOTE(kamo): Calculate in double precision
            enhanced = [
                wpe_one_iteration(
                    data.contiguous().double(),
                    p.double(),
                    taps=self.taps,
                    delay=self.delay,
                    inverse_power=self.inverse_power,
                )
                for p in power
            ]
            enhanced = [
                enh.to(dtype=data.dtype).masked_fill(make_pad_mask(ilens, enh.real), 0)
                for enh in enhanced
            ]

        # (B, F, C, T) -> (B, T, C, F)
        enhanced = [enh.permute(0, 3, 2, 1) for enh in enhanced]
        if masks is not None:
            masks = (
                [m.transpose(-1, -3) for m in masks]
                if self.nmask > 1
                else masks[0].transpose(-1, -3)
            )
        if self.nmask == 1:
            enhanced = enhanced[0]

        return enhanced, ilens, masks, power

    def predict_mask(
        self, data: ComplexTensor, ilens: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Predict mask for WPE dereverberation.
        Args:
            data (ComplexTensor): (B, T, C, F), double precision
            ilens (torch.Tensor): (B,)
        Returns:
            masks (torch.Tensor or List[torch.Tensor]): (B, T, C, F)
            ilens (torch.Tensor): (B,)
        """
        if self.use_dnn_mask:
            masks, ilens = self.mask_est(data.permute(0, 3, 2, 1).float(), ilens)
            # (B, F, C, T) -> (B, T, C, F)
            masks = [m.transpose(-1, -3) for m in masks]
            if self.nmask == 1:
                masks = masks[0]
        else:
            masks = None
        return masks, ilens

