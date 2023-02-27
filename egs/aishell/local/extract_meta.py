"""
Compute FBank feature for aishell using torchaudio.
"""

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
    'train': 120098,
    'dev': 14326,
    'test': 7176
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_data", type=str, default="/data/AISHELL-1/wav",
                        help="Directory to source audio files, "
                        f"expect subset: {', '.join(prepare_sets)} in the directory.")
    parser.add_argument("transcript", type=str, default="/data/transcript/aishell_transcript_v0.8.txt",
                        help="Path to the transcript file.")
    parser.add_argument("--subset", type=str, nargs='*',
                        choices=prepare_sets, help=f"Specify datasets in {prepare_sets}")
    parser.add_argument("--speed-perturbation", type=float, dest='sp',
                        nargs='*', default=[], help=f"Add speed perturbation to subset: {', '.join(prepare_sets)}")
    args = parser.parse_args()

    assert os.path.isfile(
        args.transcript), f"Trancript: '{args.transcript}' does not exist."
    assert os.path.isdir(args.src_data)
    if args.subset is not None:
        for _set in args.subset:
            assert _set in prepare_sets, f"--subset {_set} not in predefined datasets: {prepare_sets}"
        prepare_sets = args.subset

    for _set in prepare_sets:
        assert os.path.isdir(os.path.join(args.src_data, _set))
    for _sp_factor in args.sp:
        assert (isinstance(_sp_factor, float) or isinstance(_sp_factor, int)) and _sp_factor > 0, \
            f"Unsupport speed pertubation value: {_sp_factor}"

    trans = {}      # type: Dict[str, str]
    with open(args.transcript, 'r') as fi:
        for line in fi:
            uid, utt = line.strip().split(maxsplit=1)
            trans[uid] = utt

    audios = {}     # type: Dict[str, List[Tuple[str, str]]]
    subtrans = {}   # type: Dict[str, List[Tuple[str, str]]]

    for _set in prepare_sets:
        d_audio = os.path.join(args.src_data, _set)
        _audios = glob.glob(f"{d_audio}/**/*.wav")
        audios[_set] = []
        subtrans[_set] = []
        for _raw_wav in _audios:
            uid = os.path.basename(_raw_wav)
            if uid.endswith('.wav'):
                uid = uid[:-4]
            if uid not in trans:
                continue
            audios[_set].append((uid, _raw_wav))
            subtrans[_set].append((uid, trans[uid]))
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
