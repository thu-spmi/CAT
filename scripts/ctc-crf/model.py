import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.parameter import Parameter
import six
import numpy as np
import math
from maskedbatchnorm1d import MaskedBatchNorm1d


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


class LSTMrowCONV(torch.nn.Module):
    def __init__(self, idim,  hdim, n_layers, dropout):
        super(LSTMrowCONV, self).__init__()
        self.LSTM = LSTM(idim, hdim, n_layers, dropout)
        self.Lookahead = Lookahead(hdim,context=5)

    def forward(self, xs_pad, ilens):
        xs_pad, ilens = self.LSTM(xs_pad, ilens)
        xs_pad = self.Lookahead(xs_pad)
        return xs_pad, ilens


class TDNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, half_context=1):
        super(TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.half_context = half_context
        self.conv = torch.nn.Conv1d(self.input_dim, self.output_dim, 2*half_context+1, padding=half_context)
        self.bn = MaskedBatchNorm1d(self.output_dim, eps=1e-5, affine=True)

    def forward(self, features, input_lengths):
        tdnn_in = features.transpose(1,2)
        tdnn_out = self.conv(tdnn_in)
        output = F.relu(tdnn_out)
        output = self.bn(output, input_lengths)
        return output.transpose(1,2)


class TDNN_LSTM(torch.nn.Module):
    def __init__(self, idim,  hdim, n_layers, dropout):
        super(TDNN_LSTM, self).__init__()
        setattr(self, "tdnn0" , TDNN(idim, hdim))
        for i in six.moves.range(n_layers):
            setattr(self, "tdnn%d-1" % i, TDNN(hdim, hdim))
            setattr(self, "tdnn%d-2" % i, TDNN(hdim, hdim))
            setattr(self, "lstm%d" % i, torch.nn.LSTM(hdim,hdim, num_layers=1, bidirectional=False, batch_first=True))
            setattr(self, "bn%d" % i, MaskedBatchNorm1d(hdim, eps=1e-5, affine=True))
            setattr(self, "dropout%d" % i, torch.nn.Dropout(dropout))
        self.n_layers = n_layers

    def forward(self, xs_pad, ilens):
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        tdnn = getattr(self, 'tdnn0')
        xs_pad = tdnn(xs_pad, ilens.cuda())

        for layer in six.moves.range(self.n_layers):
            tdnn = getattr(self, 'tdnn' + str(layer)+'-1')
            xs_pad = tdnn(xs_pad, ilens.cuda())
            tdnn = getattr(self, 'tdnn' + str(layer)+'-2')
            xs_pad = tdnn(xs_pad, ilens.cuda())

            unilstm = getattr(self, 'lstm' + str(layer))
            unilstm.flatten_parameters()

            packed_input = torch.nn.utils.rnn.pack_padded_sequence(xs_pad, ilens, batch_first=True)
            packed_output, _ = unilstm(packed_input, None)
            xs_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

            cur_bn = getattr(self, 'bn'+str(layer))
            xs_pad = (cur_bn(xs_pad.transpose(1,2), ilens.cuda())).transpose(1,2)
            cur_dropout = getattr(self, 'dropout'+str(layer))
            xs_pad = cur_dropout(xs_pad)
        return xs_pad, ilens


class BLSTMN(torch.nn.Module):
    def __init__(self, idim, hdim, n_layers, dropout):
        super(BLSTMN, self).__init__()
        for i in six.moves.range(n_layers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim * 2
            setattr(self, "lstm%d" % i, torch.nn.LSTM(inputdim, hdim,num_layers=1, bidirectional=True, batch_first=True))
            setattr(self, "bn%d" % i, MaskedBatchNorm1d(hdim*2, eps=1e-5, affine=True))
            setattr(self, "dropout%d" % i, torch.nn.Dropout(dropout))
        self.n_layers = n_layers
        self.hdim = hdim

    def forward(self, xs_pad, ilens):
        for layer in six.moves.range(self.n_layers):
            lstm = getattr(self, 'lstm' + str(layer))
            lstm.flatten_parameters()
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(xs_pad, ilens,batch_first=True)
            packed_output, _ = lstm(packed_input, None)
            xs_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            cur_bn = getattr(self, 'bn'+str(layer))
            xs_pad = (cur_bn(xs_pad.transpose(1,2), ilens.cuda())).transpose(1,2)
            cur_dropout = getattr(self, 'dropout'+str(layer))
            xs_pad = cur_dropout(xs_pad)
        return xs_pad, ilens  # x: utt list of frame x dim
