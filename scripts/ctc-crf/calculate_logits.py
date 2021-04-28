"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Hongyu Xiang, Keyu An, Zheng Huahuan
"""

import json
import utils
import argparse
import kaldi_io
import numpy as np
from tqdm import tqdm
from train import build_model

import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="inference network")
    parser.add_argument("--input_scp", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--config", type=str, default=None, metavar='PATH',
                        help="Path to configuration file of backbone.")

    parser.add_argument("--nj", type=int)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to location of checkpoint.")
    args = parser.parse_args()

    assert args.resume is not None
    print("Load whole model")
    with open(args.config, 'r') as fi:
        configures = json.load(fi)

    model = build_model(args, configures, train=False)

    if torch.cuda.is_available():
        device = "cuda:0"
        print("Using GPU0.")
    else:
        device = "cpu"
        print("Using CPU.")

    model = model.to(device)
    model.load_state_dict(torch.load(args.resume, map_location=device))
    print(f"Model loaded from checkpoint: {args.resume}.")
    print("Model size:{:.2f}M".format(utils.count_parameters(model)/1e6))

    model.eval()
    n_jobs = args.nj
    writers = []

    for i in range(n_jobs):
        writers.append(
            open('{}/decode.{}.ark'.format(args.output_dir, i + 1),
                 'wb'))

    with open(args.input_scp) as f:
        lines = f.readlines()

    with torch.no_grad():
        for i, line in enumerate(tqdm(lines)):
            utt, feature_path = line.split()
            feature = np.array(kaldi_io.read_mat(feature_path))
            input_lengths = torch.IntTensor([feature.shape[0]])
            feature = torch.from_numpy(feature[None])
            feature = feature.cuda()

            netout, _ = model.forward(feature, input_lengths)
            r = netout.cpu().data.numpy()
            r[r == -np.inf] = -1e16
            r = r[0]
            kaldi_io.write_mat(writers[i % n_jobs], r, key=utt)

    for i in range(n_jobs):
        writers[i].close()
