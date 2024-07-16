# Copyright 2020 Tsinghua SPMI Lab 
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com)
#
# Description:
#   This script computes FBank features for AISHELL-4 using torchaudio. It prepares data subsets, handles 
#   speed perturbation, and validates the existence of necessary files. The key steps include parsing 
#   command-line arguments, reading text and wav.scp files, and calling data preparation functions.

import os
import glob
import argparse
from typing import List, Dict, Any, Tuple

# fmt: off
import sys
try:
    import utils.data
except ModuleNotFoundError:
    sys.path.append(".")
from utils.data import data_prep
# fmt: on


prepare_sets = [
    'train',
    'dev',
    'test'
]

expect_len = {
    'train': 70331,
    'dev': 3701,
    'test': 8708
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_data", type=str, default="./data/src",
                        help="Directory to datafiles, "
                        f"expect subset: {', '.join(prepare_sets)} in the directory.")
    parser.add_argument("--subset", type=str, nargs='*',
                        choices=prepare_sets, help=f"Specify datasets in {prepare_sets}")
    parser.add_argument("--speed-perturbation", type=float, dest='sp',
                        nargs='*', default=[], help=f"Add speed perturbation to subset: {', '.join(prepare_sets)}")
    args = parser.parse_args()

    
    assert os.path.isdir(args.src_data),f"Src does not exist."
    if args.subset is not None:
        for _set in args.subset:
            assert _set in prepare_sets, f"--subset {_set} not in predefined datasets: {prepare_sets}"
        prepare_sets = args.subset

    for _set in prepare_sets:
        assert os.path.isdir(os.path.join(args.src_data, _set))
        assert os.path.isfile(
            os.path.join(args.src_data, _set, "text")), f"Text: '{_set}'text does not exist."
    
    for _sp_factor in args.sp:
        assert (isinstance(_sp_factor, float) or isinstance(_sp_factor, int)) and _sp_factor > 0, \
            f"Unsupport speed pertubation value: {_sp_factor}"


    audios = {}     # type: Dict[str, List[Tuple[str, str]]]
    subtrans = {}   # type: Dict[str, List[Tuple[str, str]]]

    for _set in prepare_sets:
        
        text = {}      # type: Dict[str, str]
        wave = {}      # type: Dict[str, str]
        text_path = os.path.join(args.src_data, _set, "text")
        wavescp_path = os.path.join(args.src_data, _set, "wav.scp")
        with open(text_path, 'r') as fi:
            for line in fi:
                uid, utt = line.strip().split(maxsplit=1)
                text[uid] = utt
        
        with open(wavescp_path, 'r') as scp:
            for line in scp:
                uid1, wavpath = line.strip().split(maxsplit=1)
                wave[uid1] = wavpath
        
        
        audios[_set] = []
        subtrans[_set] = []
        for uid, _ in text.items():
            if uid not in wave:
                continue
            audios[_set].append((uid, wave[uid]))
            subtrans[_set].append((uid, text[uid]))
        if len(audios[_set]) != expect_len[_set]:
            sys.stderr.write(
                f"warning: found {len(audios[_set])} audios in {_set} subset, but expected {expect_len[_set]}\n")

    data_prep.prepare_kaldi_feat(
        subsets=prepare_sets,
        trans=subtrans,
        audios=audios,
        num_mel_bins=80,
        apply_cmvn=False,
        speed_perturb=args.sp
    )
