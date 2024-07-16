# Copyright 2020 Tsinghua SPMI Lab 
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com)
#
# Description:
#   This script defines the ChannelSelector class, which is a neural network module for selecting a specific channel from multi-channel input tensors.

import torch.nn as nn

class ChannelSelector(nn.Module):
    """
    A neural network module for selecting a specific channel from multi-channel input tensors.

    Args:
        total_channels (int): Total number of channels in the input tensor.
        chosen_channel (int): Index of the channel to be selected.

    Methods:
        forward(x: torch.Tensor, len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
            Selects the specified channel from the input tensor.
    """
    def __init__(self, total_channels,chosen_channel):
        super(ChannelSelector, self).__init__()
        if not chosen_channel < total_channels:
            raise ValueError("Chosen channel index out of bounds")
        self.total_channels = total_channels
        self.chosen_channel = chosen_channel

    def forward(self, x, len):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, F, 2).
            len (torch.Tensor): Tensor representing the lengths of the sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: 
                - Selected channel tensor of shape (B, T, F, 2).
                - Tensor representing the lengths of the sequences.
                - Index of the chosen channel.
        """
        # x: input tensor of shape (B, T, C, F, 2)
        return  x[:, :, self.chosen_channel, :, :] ,len,self.chosen_channel
