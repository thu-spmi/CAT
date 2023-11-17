import torch.nn as nn

class ChannelSelector(nn.Module):
    def __init__(self, total_channels,chosen_channel):
        super(ChannelSelector, self).__init__()
        if not chosen_channel < total_channels:
            raise ValueError("Chosen channel index out of bounds")
        self.total_channels = total_channels
        self.chosen_channel = chosen_channel

    def forward(self, x, len):
        # x: input tensor of shape (B, T, C, F, 2)
        return  x[:, :, self.chosen_channel, :, :] ,len,self.chosen_channel
