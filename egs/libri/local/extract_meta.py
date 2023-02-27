"""
Compute FBank feature for librispeech-960 using torchaudio.
"""


import os
import sys
import glob
import argparse
from typing import List, Dict, Tuple

prepare_sets = [
    'dev-clean',
    'dev-other',
    'test-clean',
    'test-other',
    'train-clean-100',
    'train-clean-360',
    'train-other-500'
]

expect_len = {
    'dev-clean': 2703,
    'dev-other': 2864,
    'test-clean': 2620,
    'test-other': 2939,
    'train-clean-100': 28539,
    'train-clean-360': 104014,
    'train-other-500': 148688
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_data", type=str, default="/data/librispeech/LibriSpeech",
                        help="Directory to source audio files, "
                        f"expect sub-dir: {', '.join(prepare_sets)} in the directory.")

    parser.add_argument("--subset", type=str, nargs='*', default=None,
                        choices=prepare_sets, help="Subset to be processes, default all.")
    parser.add_argument("--speed-perturbation", type=float, dest='sp',
                        nargs='*', default=[], help=f"Add speed perturbation to subset: {', '.join(prepare_sets)}")
    args = parser.parse_args()

    process_sets = args.subset
    if process_sets is None:
        process_sets = prepare_sets

    assert os.path.isdir(args.src_data)
    for _set in process_sets:
        assert os.path.isdir(os.path.join(args.src_data, _set)
                             ), f"subset '{_set}' not found in {args.src_data}"

    trans = {}      # type: Dict[str, List[Tuple[str, str]]]
    audios = {}     # type: Dict[str, List[Tuple[str, str]]]
    for _set in process_sets:
        d_audio = os.path.join(args.src_data, _set)
        _audios = glob.glob(f"{d_audio}/**/**/*.flac")
        trans[_set] = []
        for f_ in sorted(glob.glob(f"{d_audio}/**/**/*.trans.txt")):
            with open(f_, 'r') as fi:
                for line in fi:
                    uid, utt = line.strip().split(maxsplit=1)
                    trans[_set].append((uid, utt))

        audios[_set] = []
        for _raw_wav in _audios:
            uid = os.path.basename(_raw_wav)
            if uid.endswith('.flac'):
                uid = uid[:-4]
            audios[_set].append((uid, _raw_wav))

        assert len(audios[_set]) == len(trans[_set]), \
            f"# audio mismatches # transcript in {_set}: {len(audios[_set])} != {len(trans[_set])}"
        if len(audios[_set]) != expect_len[_set]:
            sys.stderr.write(
                f"warning: found {len(audios[_set])} audios in {_set} subset, but expected {expect_len[_set]}")

    from cat.utils.data import data_prep
    data_prep.prepare_kaldi_feat(
        subsets=process_sets,
        trans=trans,
        audios=audios,
        num_mel_bins=80,
        speed_perturb=args.sp
    )
