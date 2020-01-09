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
from torch.autograd import Variable
from model import BLSTM 
os.environ['CUDA_VISIBLE_DEVICES']='2'

class Model(nn.Module):
    def __init__(self, idim, hdim, K, n_layers, dropout):
        super(Model, self).__init__()
        self.net = BLSTM(idim, hdim, n_layers, dropout)
        self.linear = nn.Linear(hdim * 2, K)

    def forward(self, logits, input_lengths):
        netout, _ = self.net(logits, input_lengths)
        netout = self.linear(netout)
        netout = F.log_softmax(netout, dim=2)

        return netout

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="inference network")
    parser.add_argument("--input_scp", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_unit", type=int)
    parser.add_argument("--hdim",type=int,default=512)
    parser.add_argument("--layers",type=int,default=6)
    parser.add_argument("--dropout",type=int,default=0.5)
    parser.add_argument("--feature_size",type=int,default=120)
    parser.add_argument("--data_path",type=str)
    parser.add_argument("--nj",type=int)
    args = parser.parse_args()

    batch_size = 128

    model = Model(args.feature_size, args.hdim, args.output_unit, args.layers, args.dropout)
    model.load_state_dict(torch.load(args.data_path+'/models/best_model'))
    model.eval()
    model.cuda()
    n_jobs = args.nj
    writers = []
    write_mode = 'w'
    if sys.version > '3':
        write_mode = 'wb'
        
    for i in range(n_jobs):
        writers.append(open('{}/decode.{}.ark'.format(args.output_dir, i+1), write_mode))

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
        kaldi_io.write_mat(writers[i%n_jobs], r, key=utt)

    for i in range(n_jobs):
        writers[i].close()
