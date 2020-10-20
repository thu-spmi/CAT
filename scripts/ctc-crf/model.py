import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
from maskedbatchnorm1d import MaskedBatchNorm1d
import six
from ctc_crf import CTC_CRF_LOSS, WARP_CTC_LOSS

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
        kernel_size = 3
        padding = 1
        self.conv1_1 = torch.nn.Conv2d(
            in_channel, 64, kernel_size, stride=1, padding=padding)
        self.conv1_2 = torch.nn.Conv2d(
            64, 64, kernel_size, stride=1, padding=padding)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2_1 = torch.nn.Conv2d(
            64, 128, kernel_size, stride=1, padding=padding)
        self.conv2_2 = torch.nn.Conv2d(
            128, 128, kernel_size, stride=1, padding=padding)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.in_channel = in_channel

    def forward(self, xs_pad, ilens):
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                             xs_pad.size(2) // self.in_channel).transpose(1, 2)
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = self.bn1(xs_pad)
        xs_pad = F.max_pool2d(xs_pad, [1, 2], stride=[1, 2], ceil_mode=True)
        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = self.bn2(xs_pad)
        xs_pad = F.max_pool2d(xs_pad, [1, 2], stride=[1, 2], ceil_mode=True)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        return xs_pad, ilens


class BLSTM(nn.Module):
    def __init__(self, idim,  hdim, n_layers, dropout):
        super(BLSTM, self).__init__()
        self.lstm1 = nn.LSTM(idim, hdim, num_layers=n_layers,
                             bidirectional=True, batch_first=True, dropout=dropout)

    def forward(self, features, input_lengths, hidden=None):
        self.lstm1.flatten_parameters()
        total_length = features.size(1)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            features, input_lengths, batch_first=True)
        packed_output, _ = self.lstm1(packed_input, hidden)
        lstm_out, ilens = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)
        return lstm_out, ilens


class LSTM(nn.Module):
    def __init__(self, idim, hdim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(idim, hdim, num_layers=n_layers,
                             bidirectional=False, batch_first=True, dropout=dropout)

    def forward(self, features, input_lengths, hidden=None):
        self.lstm1.flatten_parameters()
        total_length = features.size(1)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            features, input_lengths, batch_first=True)
        packed_output, _ = self.lstm1(packed_input, hidden)
        lstm_out, ilens = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)
        return lstm_out, ilens


class VGGBLSTM(torch.nn.Module):
    def __init__(self, idim, hdim, n_layers, dropout, in_channel=3):
        super(VGGBLSTM, self).__init__()

        self.VGG = VGG2L(in_channel)
        self.BLSTM = BLSTM(get_vgg2l_odim(idim, in_channel=in_channel),
                           hdim, n_layers, dropout)

    def forward(self, xs_pad, ilens):
        xs_pad, ilens = self.VGG(xs_pad, ilens)
        xs_pad, ilens = self.BLSTM(xs_pad, ilens)
        return xs_pad, ilens


class VGGLSTM(torch.nn.Module):
    def __init__(self, idim,  hdim, n_layers, dropout, in_channel=3):
        super(VGGLSTM, self).__init__()
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
        self.Lookahead = Lookahead(hdim, context=5)

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
        self.conv = torch.nn.Conv1d(
            self.input_dim, self.output_dim, 2*half_context+1, padding=half_context)
        self.bn = MaskedBatchNorm1d(self.output_dim, eps=1e-5, affine=True)

    def forward(self, features, input_lengths):
        tdnn_in = features.transpose(1, 2)
        tdnn_out = self.conv(tdnn_in)
        output = F.relu(tdnn_out)
        output = self.bn(output, input_lengths)
        return output.transpose(1, 2)

class TDNN_downsample(torch.nn.Module):
    def __init__(self, idim, hdim, n_layers=7,dropout=0.5):
        super(TDNN_downsample, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.n_layers = n_layers
        self.dropout = dropout
        setattr(self,"conv0" , torch.nn.Conv1d(self.idim, self.hdim, 5, padding=2))
        setattr(self, "dropout0" , torch.nn.Dropout(dropout))
        setattr(self,"conv1" , torch.nn.Conv1d(self.hdim, self.hdim, 5, dilation=2,padding=4))
        setattr(self, "dropout1" , torch.nn.Dropout(dropout))
        setattr(self,"conv2" , torch.nn.Conv1d(self.hdim, self.hdim, 5, padding=2))
        setattr(self, "dropout2" , torch.nn.Dropout(dropout))
        setattr(self,"conv3" , torch.nn.Conv1d(self.hdim, self.hdim, 3, stride=3))
        setattr(self, "dropout3" , torch.nn.Dropout(dropout))
        setattr(self,"conv4" , torch.nn.Conv1d(self.hdim, self.hdim, 5, dilation=2,padding=4))
        setattr(self, "dropout4" , torch.nn.Dropout(dropout))
        setattr(self,"conv5" , torch.nn.Conv1d(self.hdim, self.hdim, 5, padding=2))
        setattr(self, "dropout5" , torch.nn.Dropout(dropout))
        setattr(self,"conv6" , torch.nn.Conv1d(self.hdim, self.hdim, 5, dilation=2,padding=4))

    def forward(self, features, input_lengths):
        for layer in six.moves.range(self.n_layers):
            conv= getattr(self, 'conv' + str(layer))
            features = features.transpose(1,2)
            features = conv(features)
            features = F.relu(features)
            features = features.transpose(1,2)
            features = F.layer_norm(features, [features.size()[-1]])
            if layer < self.n_layers-1:
               dropout = getattr(self, 'dropout' + str(layer))
               features = dropout(features)
        return features,input_lengths

class TDNN_LSTM(torch.nn.Module):
    def __init__(self, idim,  hdim, n_layers, dropout):
        super(TDNN_LSTM, self).__init__()
        setattr(self, "tdnn0", TDNN(idim, hdim))
        for i in six.moves.range(n_layers):
            setattr(self, "tdnn%d-1" % i, TDNN(hdim, hdim))
            setattr(self, "tdnn%d-2" % i, TDNN(hdim, hdim))
            setattr(self, "lstm%d" % i, torch.nn.LSTM(
                hdim, hdim, num_layers=1, bidirectional=False, batch_first=True))
            setattr(self, "bn%d" % i, MaskedBatchNorm1d(
                hdim, eps=1e-5, affine=True))
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

            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                xs_pad, ilens, batch_first=True)
            packed_output, _ = unilstm(packed_input, None)
            xs_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True)

            cur_bn = getattr(self, 'bn'+str(layer))
            xs_pad = (cur_bn(xs_pad.transpose(1, 2),
                             ilens.cuda())).transpose(1, 2)
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
            setattr(self, "lstm%d" % i, torch.nn.LSTM(inputdim, hdim,
                                                      num_layers=1, bidirectional=True, batch_first=True))
            setattr(self, "bn%d" % i, MaskedBatchNorm1d(
                hdim*2, eps=1e-5, affine=True))
            setattr(self, "dropout%d" % i, torch.nn.Dropout(dropout))
        self.n_layers = n_layers
        self.hdim = hdim

    def forward(self, xs_pad, ilens):
        for layer in six.moves.range(self.n_layers):
            lstm = getattr(self, 'lstm' + str(layer))
            lstm.flatten_parameters()
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                xs_pad, ilens, batch_first=True)
            packed_output, _ = lstm(packed_input, None)
            xs_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True)
            cur_bn = getattr(self, 'bn'+str(layer))
            xs_pad = (cur_bn(xs_pad.transpose(1, 2),
                             ilens.cuda())).transpose(1, 2)
            cur_dropout = getattr(self, 'dropout'+str(layer))
            xs_pad = cur_dropout(xs_pad)
        return xs_pad, ilens  # x: utt list of frame x dim


class ChunkBLSTM(nn.Module):
    def __init__(self, idim,  hdim, dropout):
        super(ChunkBLSTM, self).__init__()
        self.lstmbase = nn.LSTM(
            idim, hdim, 3, bidirectional=True, batch_first=True, dropout=dropout)
        self.lstm1 = nn.LSTM(
            hdim*2, hdim, 1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(
            hdim*2, hdim, 1, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(
            hdim*2, hdim, 1, bidirectional=True, batch_first=True)
        self.dropoutbase = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features, chunk_size, mode, hiddenbase=None, hidden1=None, hidden2=None, hidden3=None):
        self.lstmbase.flatten_parameters()
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()

        N_chunks = features.size(1)//chunk_size
        if mode == 'train':
            x = features.view(features.size(0)*N_chunks,
                              chunk_size, features.size(2))
        else:
            x = features
        if mode == 'eval':
            (hbase, cbase) = hiddenbase
            (h1, c1) = hidden1
            (h2, c2) = hidden2
            (h3, c3) = hidden3
            if hbase is None or cbase is None:
                hiddenbase = None
            if h1 is None or c1 is None:
                hidden1 = None
            if h2 is None or c2 is None:
                hidden2 = None
            if h3 is None or c3 is None:
                hidden3 = None

        lstm_outbase, (h_base, c_base) = self.lstmbase(x, hiddenbase)
        lstm_out_1, (h1, c1) = self.lstm1(
            self.dropoutbase(lstm_outbase), hidden1)
        lstm_out_2, (h2, c2) = self.lstm2(self.dropout1(lstm_out_1), hidden2)
        lstm_out_3, (h3, c3) = self.lstm3(self.dropout2(lstm_out_2), hidden3)
        if mode == 'train':
            out1 = lstm_out_1.contiguous().view(lstm_out_1.size(
                0)//N_chunks, lstm_out_1.size(1)*N_chunks, -1)
            out2 = lstm_out_2.contiguous().view(lstm_out_2.size(
                0)//N_chunks, lstm_out_2.size(1)*N_chunks, -1)
            out3 = lstm_out_3.contiguous().view(lstm_out_3.size(
                0)//N_chunks, lstm_out_3.size(1)*N_chunks, -1)
            return h_base, c_base, out1, h1, c1, out2, h2, c2, out3, h3, c3
        else:
            return h_base, c_base, lstm_out_1.contiguous(), h1, c1, lstm_out_2.contiguous(), h2, c2, lstm_out_3.contiguous(), h3, c3


class BLSTM_REG(nn.Module):
    def __init__(self, idim,  hdim, dropout):
        super(BLSTM_REG, self).__init__()
        self.lstmbase = nn.LSTM(
            idim, hdim, 3, bidirectional=True, batch_first=True, dropout=dropout)
        self.lstm1 = nn.LSTM(
            hdim*2, hdim, 1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(
            hdim*2, hdim, 1, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(
            hdim*2, hdim, 1, bidirectional=True, batch_first=True)
        self.dropoutbase = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features, hiddenbase=None, hidden1=None, hidden2=None, hidden3=None):
        self.lstmbase.flatten_parameters()
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
        outbase, (h_base, c_base) = self.lstmbase(features, hiddenbase)
        out1, (h1, c1) = self.lstm1(self.dropoutbase(outbase), hidden1)
        out2, (h2, c2) = self.lstm2(self.dropout1(out1), hidden2)
        out3, (h3, c3) = self.lstm3(self.dropout2(out2), hidden3)
        return out1, h1, c1, out2, h2, c2, out3, h3, c3


class BLSTM_REG_2(torch.nn.Module):
    def __init__(self, idim, hdim,  dropout):
        super(BLSTM_REG_2, self).__init__()
        for i in six.moves.range(6):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim * 2
            setattr(self, "lstm%d" % i, torch.nn.LSTM(inputdim, hdim,
                                                      num_layers=1, bidirectional=True, batch_first=True))
            setattr(self, "dropout%d" % i, torch.nn.Dropout(dropout))
        self.n_layers = 6
        self.hdim = hdim

    def forward(self, xs_pad, ilens):
        for layer in six.moves.range(self.n_layers):
            lstm = getattr(self, 'lstm' + str(layer))
            lstm.flatten_parameters()
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                xs_pad, ilens, batch_first=True)
            packed_output, (h, c) = lstm(packed_input, None)
            xs_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True)
            if layer == 3:
                out1 = xs_pad
                h1 = h
                c1 = c
            if layer == 4:
                out2 = xs_pad
                h2 = h
                c2 = c
            if layer == 5:
                out3 = xs_pad
                h3 = h
                c3 = c
            cur_dropout = getattr(self, 'dropout'+str(layer))
            xs_pad = cur_dropout(xs_pad)
        return out1, h1, c1, out2, h2, c2, out3, h3, c3


class ChunkBLSTM_with_Context(nn.Module):
    def __init__(self, idim,  hdim, dropout, context):
        super(ChunkBLSTM_with_Context, self).__init__()
        self.lstmbase = nn.LSTM(
            idim, hdim, 3, bidirectional=True, batch_first=True, dropout=dropout)
        self.lstm1 = nn.LSTM(
            hdim*2, hdim, 1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(
            hdim*2, hdim, 1, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(
            hdim*2, hdim, 1, bidirectional=True, batch_first=True)
        self.dropoutbase = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.context = context

    def forward(self, features, chunk_size):
        self.lstmbase.flatten_parameters()
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()

        N_chunks = features.size(1)//chunk_size
        # [B*N_chunks] * chunk_size *120
        x = features.view(features.size(0)*N_chunks,
                          chunk_size, features.size(2))

        right_context = torch.zeros(x.size()[0], self.context, x.size()[2])
        left_context = torch.zeros(x.size()[0], self.context, x.size()[2])

        right_context[:-1, :, :] = x[1:, :self.context, :]
        left_context[1:, :, :] = x[:-1, -self.context:, :]
        x_with_context = torch.cat(
            (left_context.cpu(), x.cpu(), right_context.cpu()), dim=1)
        x_with_context[0::N_chunks, :self.context, :] = 0
        x_with_context[N_chunks-1::N_chunks, -self.context:, :] = 0

        lstm_outbase_with_context, _ = self.lstmbase(x_with_context.cuda())
        lstm_out_1_with_context, _ = self.lstm1(
            self.dropoutbase(lstm_outbase_with_context))
        lstm_out_2_with_context, _ = self.lstm2(
            self.dropout1(lstm_out_1_with_context))
        lstm_out_3_with_context, _ = self.lstm3(
            self.dropout2(lstm_out_2_with_context))

        lstm_out_1 = lstm_out_1_with_context[:, self.context:-self.context, :]
        lstm_out_2 = lstm_out_2_with_context[:, self.context:-self.context, :]
        lstm_out_3 = lstm_out_3_with_context[:, self.context:-self.context, :]

        out1 = lstm_out_1.contiguous().view(lstm_out_1.size(
            0)//N_chunks, lstm_out_1.size(1)*N_chunks, -1)
        out2 = lstm_out_2.contiguous().view(lstm_out_2.size(
            0)//N_chunks, lstm_out_2.size(1)*N_chunks, -1)
        out3 = lstm_out_3.contiguous().view(lstm_out_3.size(
            0)//N_chunks, lstm_out_3.size(1)*N_chunks, -1)
        return out1, out2, out3


class CAT_Model(nn.Module):
    def __init__(self, net, idim, hdim, K, n_layers, dropout, lamb, use_ctc_crf=False):
        super(CAT_Model, self).__init__()
        self.net = eval(net)(idim, hdim, n_layers, dropout=dropout)
        if net in ['BLSTM', 'BLSTMN']:
            self.linear = nn.Linear(hdim * 2, K)
        else:
            self.linear = nn.Linear(hdim, K)
        if use_ctc_crf:
            self.loss_fn = CTC_CRF_LOSS(lamb=lamb)
        else:
            self.loss_fn = WARP_CTC_LOSS()

    def forward(self, logits, labels_padded, input_lengths, label_lengths):
        # rearrange by input_lengths
        input_lengths, indices = torch.sort(input_lengths, descending=True)
        assert indices.dim() == 1, "input_lengths should have only 1 dim"
        logits = torch.index_select(logits, 0, indices)
        labels_padded = torch.index_select(labels_padded, 0, indices)
        label_lengths = torch.index_select(label_lengths, 0, indices)

        labels_padded = labels_padded.cpu()
        input_lengths = input_lengths.cpu()
        label_lengths = label_lengths.cpu()

        label_list = [
            labels_padded[i, :x] for i, x in enumerate(label_lengths)
        ]
        labels = torch.cat(label_list)
        netout, _ = self.net(logits, input_lengths)
        netout = self.linear(netout)
        netout = F.log_softmax(netout, dim=2)

        loss = self.loss_fn(netout, labels, input_lengths, label_lengths)
        return loss


class CAT_RegModel(nn.Module):
    def __init__(self, idim, hdim,  K,  dropout, lamb):
        super(CAT_RegModel, self).__init__()
        self.net = BLSTM_REG_2(idim, hdim, dropout)
        self.linear = nn.Linear(hdim*2, K)

    def forward(self, logits, labels_padded, input_lengths, label_lengths):
        # rearrange by input_lengths
        input_lengths, indices = torch.sort(input_lengths, descending=True)
        assert indices.dim() == 1, "input_lengths should have only 1 dim"
        logits = torch.index_select(logits, 0, indices)
        labels_padded = torch.index_select(labels_padded, 0, indices)
        label_lengths = torch.index_select(label_lengths, 0, indices)

        labels_padded = labels_padded.cpu()
        input_lengths = input_lengths.cpu()
        label_lengths = label_lengths.cpu()

        label_list = [labels_padded[i, :x]
                      for i, x in enumerate(label_lengths)]
        labels = torch.cat(label_list)
        out1, h1, c1, out2, h2, c2, out3, h3, c3 = self.net(
            logits, input_lengths)
        return out1, out2, out3


class CAT_Chunk_Model(nn.Module):
    def __init__(self, idim, hdim,  K,  dropout, lamb, reg_weight, use_ctc_crf=False):
        super(CAT_Chunk_Model, self).__init__()
        self.net = ChunkBLSTM_with_Context(idim, hdim, context=10, dropout=0.5)
        self.linear = nn.Linear(hdim*2, K)
        self.reg_weight = reg_weight
        self.criterion = nn.MSELoss(size_average=False)
        
        if use_ctc_crf:
            self.loss_fn = CTC_CRF_LOSS(lamb=lamb)
        else:
            self.loss_fn = WARP_CTC_LOSS()
   
    def forward(self, logits, labels_padded, input_lengths, label_lengths, chunk_size, out1_reg, out2_reg, out3_reg):
        input_lengths, indices = torch.sort(input_lengths, descending=True)
        assert indices.dim() == 1, "input_lengths should have only 1 dim"
        logits = torch.index_select(logits, 0, indices)
        labels_padded = torch.index_select(labels_padded, 0, indices)
        label_lengths = torch.index_select(label_lengths, 0, indices)

        labels_padded = labels_padded.cpu()
        input_lengths = input_lengths.cpu()
        label_lengths = label_lengths.cpu()

        label_list = [labels_padded[i, :x]
                      for i, x in enumerate(label_lengths)]
        labels = torch.cat(label_list)
        N_sample = logits.size()[0]
        out1, out2, out3 = self.net(logits, chunk_size)
        dist_1 = self.criterion(out1, out1_reg)
        dist_2 = self.criterion(out2, out2_reg)
        dist_3 = self.criterion(out3, out3_reg)
        loss_reg = (dist_1 + dist_2 + dist_3)/N_sample*self.reg_weight

        netout = self.linear(out3)
        netout = F.log_softmax(netout, dim=2)

        loss_cls = self.loss_fn(netout, labels, input_lengths, label_lengths)
        loss = loss_cls + loss_reg
        return loss, loss_cls, loss_reg
