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
import math
from torch.autograd import Variable
from model import ChunkBLSTM_with_Context

class Model(nn.Module):
    def __init__(self, idim, hdim, K, context, dropout):
        super(Model, self).__init__()
        self.net = ChunkBLSTM_with_Context(idim, hdim,context=context, dropout=dropout)
        self.linear = nn.Linear(hdim*2 , K)

    def forward(self, logits,chunk_size):
        out1,out2,out3 = self.net(logits,chunk_size)
        netout = self.linear(out3)
        netout = F.log_softmax(netout, dim=2)

        return out1,out2,out3,netout


def pad_tensor(t, pad_to_length, dim):
    pad_size = list(t.shape)
    pad_size[dim] = pad_to_length - t.size(dim)
    return torch.cat([t, torch.zeros(*pad_size).type_as(t)], dim=dim)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="inference network")
    parser.add_argument("--input_scp", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_unit", type=int)
    parser.add_argument("--hdim",type=int,default=512)
    parser.add_argument("--dropout",type=float,default=0.5)
    parser.add_argument("--feature_size",type=int,default=120)
    parser.add_argument("--model",type=str)
    parser.add_argument("--nj",type=int)
    parser.add_argument("--chunk_size",type=int,default=40)
    parser.add_argument("--context",type=int,default=10)
    args = parser.parse_args()

    model = Model(args.feature_size, args.hdim, args.output_unit, args.context, args.dropout)
    model.load_state_dict(torch.load(args.model)['model'])
    model.eval()
    model.cuda()
    n_jobs = args.nj
    writers = []
    online = True
    forward = False
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
        hdim=args.hdim
        input_length = feature.size()[1]
        netout = []
        N_chunks = math.ceil(input_length/args.chunk_size)
        input_length = args.chunk_size*N_chunks
        feature = pad_tensor(feature, input_length,1)
        feature = feature.cuda()
        out1,out2,out3,netout = model.forward(feature,input_length)  
        r = netout.cpu().data.numpy()
        r[r == -np.inf] = -1e16
        r = r[0]
        kaldi_io.write_mat(writers[i%n_jobs], r, key=utt)

    for i in range(n_jobs):
        writers[i].close()
