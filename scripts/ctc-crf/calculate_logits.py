import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import kaldi_io
import six
import argparse
import json
from torch.autograd import Variable
from model import BLSTM, LSTM, VGGBLSTM, VGGLSTM, LSTMrowCONV, TDNN_LSTM, BLSTMN


class Model(nn.Module):
    def __init__(self, net, idim, hdim, K, n_layers, dropout):
        super(Model, self).__init__()
        self.net = eval(net)(idim, hdim, n_layers, dropout)
        if net in ['BLSTM', 'BLSTMN', 'VGGBLSTM']:
            self.linear = nn.Linear(hdim * 2, K)
        else:
            self.linear = nn.Linear(hdim, K)

    def forward(self, logits, input_lengths):
        netout, _ = self.net(logits, input_lengths)
        netout = self.linear(netout)
        netout = F.log_softmax(netout, dim=2)

        return netout


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="inference network")
    parser.add_argument(
        "--arch",
        choices=[
            'BLSTM', 'LSTM', 'VGGBLSTM', 'VGGLSTM', 'LSTMrowCONV', 'TDNN_LSTM',
            'BLSTMN'
        ],
        default='BLSTM')
    parser.add_argument("--input_scp", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_unit", type=int)
    parser.add_argument("--hdim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--feature_size", type=int, default=120)
    parser.add_argument("--model", type=str)
    parser.add_argument("--nj", type=int)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as fin:
            config = json.load(fin)
            args.arch = config['arch']
            args.feature_size = config['feature_size']
            args.hdim = config['hdim']
            args.output_unit = config['output_unit']
            args.layers = config['layers']
            args.dropout = config['dropout']

    model = Model(args.arch, args.feature_size, args.hdim, args.output_unit,
                  args.layers, args.dropout)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    model.cuda()
    n_jobs = args.nj
    writers = []
    write_mode = 'w'
    if sys.version > '3':
        write_mode = 'wb'

    for i in range(n_jobs):
        writers.append(
            open('{}/decode.{}.ark'.format(args.output_dir, i + 1),
                 write_mode))

    with open(args.input_scp) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        utt, feature_path = line.split()
        feature = kaldi_io.read_mat(feature_path)
        input_lengths = torch.IntTensor([feature.shape[0]])
        feature = torch.from_numpy(feature[None])
        feature = feature.cuda()

        netout = model.forward(feature, input_lengths)
        r = netout.cpu().data.numpy()
        r[r == -np.inf] = -1e16
        r = r[0]
        kaldi_io.write_mat(writers[i % n_jobs], r, key=utt)

    for i in range(n_jobs):
        writers[i].close()
