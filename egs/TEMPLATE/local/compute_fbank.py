"""
Compute FBank feature for yesno example using torchaudio.
"""

import os
import sys
import glob
import math
import argparse
from typing import List, Dict

import kaldiio

import torch
import torchaudio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yesnodir", type=str,
                        help="Directory to yesno audio files.")
    args = parser.parse_args()

    audios = glob.glob(f"{args.yesnodir}/?_?_?_?_?_?_?_?.wav")
    if len(audios) != 60:
        print(
            f"warning: there expect to be 60 audio files, instead found {len(audios)}")
        fmtuid = r"yesno-{:0"+math.ceil(math.log10(len(audios)))+r"}"
    else:
        fmtuid = r"yesno-{:02}"

    trans = {}      # type: Dict[str, List[str]]
    specs = {}      # type: Dict[str, torch.Tensor]
    # get the sample rate of audio files
    _, sample_frequency = torchaudio.load(audios[0])
    num_mel_bins = 80
    for idx, file in enumerate(audios):
        uttid = fmtuid.format(idx)
        fname = os.path.basename(file)
        if fname.endswith('.wav'):
            fname = fname[:-4]
        trans[uttid] = fname.split('_')

        specs[uttid] = torchaudio.compliance.kaldi.fbank(
            torchaudio.load(file)[0],
            sample_frequency=sample_frequency,
            num_mel_bins=num_mel_bins)

    os.makedirs("data/src/yesno", exist_ok=True)
    # export transcript
    mapping = {'0': 'NO', '1': 'YES'}
    with open("data/src/yesno/text", 'w') as fo:
        fo.write(
            '\n'.join(f"{uttid}\t{' '.join(mapping[x] for x in seq)}"
                      for uttid, seq in trans.items())
        )

    # export spec as kaldi format
    scp_dir = "data/src/yesno"
    os.makedirs(scp_dir, exist_ok=True)
    with kaldiio.WriteHelper(f'ark,scp:{scp_dir}/feats.ark,{scp_dir}/feats.scp') as writer:
        for uid, mat in specs.items():
            writer(uid, mat.numpy())
