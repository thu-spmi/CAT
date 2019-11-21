import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.parameter import Parameter
import six
import numpy as np
import math


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

def get_vgg2l_odim(idim, in_channel=1, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels

class VGG2L(torch.nn.Module):
    def __init__(self, in_channel=4):
        super(VGG2L, self).__init__()
        kernel_size=3
        padding=1
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, kernel_size, stride=1, padding=padding)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size, stride=1, padding=padding)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size, stride=1, padding=padding)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size, stride=1, padding=padding)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.in_channel = in_channel

    def forward(self, xs_pad, ilens):
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                             xs_pad.size(2) // self.in_channel).transpose(1, 2)
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = self.bn1(xs_pad)
        xs_pad = F.max_pool2d(xs_pad,[1,2], stride=[1,2], ceil_mode=True)
        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = self.bn2(xs_pad)
        xs_pad = F.max_pool2d(xs_pad, [1,2], stride=[1,2], ceil_mode=True)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        return xs_pad, ilens


class BLSTM(nn.Module):
    def __init__(self, idim,  hdim, n_layers, dropout):
        super(BLSTM, self).__init__()
        self.lstm1 = nn.LSTM(idim, hdim, num_layers=n_layers, bidirectional=True, batch_first=True, dropout=dropout)

    def forward(self, features, input_lengths, hidden=None):
        self.lstm1.flatten_parameters()
        total_length = features.size(1)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(features, input_lengths, batch_first=True)
        packed_output, _ = self.lstm1(packed_input, hidden)
        lstm_out, ilens  = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return lstm_out, ilens


class LSTM(nn.Module):
    def __init__(self, idim, hdim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(idim, hdim, num_layers=n_layers, bidirectional=False, batch_first=True, dropout=dropout)

    def forward(self, features, input_lengths, hidden=None):
        self.lstm1.flatten_parameters()
        total_length = features.size(1)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(features, input_lengths, batch_first=True)
        packed_output, _ = self.lstm1(packed_input, hidden)
        lstm_out, ilens  = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return lstm_out, ilens


class VGGBLSTM(torch.nn.Module):
    def __init__(self, idim, hdim, n_layers, dropout, in_channel=3):
        super(EncoderBLSTM, self).__init__()

        self.VGG = VGG2L(in_channel)
        self.BLSTM = BLSTM(get_vgg2l_odim(idim, in_channel=in_channel),
                                hdim, n_layers, dropout)

    def forward(self, xs_pad, ilens):
        xs_pad, ilens = self.VGG(xs_pad, ilens)
        xs_pad, ilens = self.BLSTM(xs_pad, ilens)
        return xs_pad, ilens


class VGGLSTM(torch.nn.Module):
    def __init__(self, idim,  hdim, n_layers, dropout, in_channel=3):
        super(EncoderLSTM, self).__init__()
        self.VGG = VGG2L(in_channel)
        self.LSTM = LSTM(get_vgg2l_odim(idim, in_channel=in_channel),
                                hdim, n_layers, dropout)

    def forward(self, xs_pad, ilens):
        xs_pad, ilens = self.VGG(xs_pad, ilens)
        xs_pad, ilens = self.LSTM(xs_pad, ilens)
        return xs_pad, ilens


class Lookahead(nn.Module):
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1,
                              groups=self.n_features, padding=0, bias=None)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        return x


class LSTMrowCONV(torch.nn.Module)
    def __init__(self, idim,  hdim, n_layers, dropout):
        super(LSTMrowCONV, self).__init__()
        self.LSTM = LSTM(idim, hdim, n_layers, dropout)
        self.Lookahead = Lookahead(hdim,context=5)

    def forward(self, xs_pad, ilens):
        xs_pad, ilens = self.LSTM(xs_pad, ilens)
        xs_pad = self.Lookahead(xs_pad)
        return xs_pad, ilens


