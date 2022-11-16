"""
Compute FBank feature for commonvoice format source data. using torchaudio.
"""


import os
import sys
import argparse
from typing import List, Dict, Any, Tuple


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_data", type=str, default="data/src", nargs='?',
                        help="Directory includes the meta infos.")
    parser.add_argument("--subset", type=str, required=True, nargs='*',
                        help=f"Subse(s) to be processed.")
    parser.add_argument("--cmvn", action="store_true", default=False,
                        help="Apply CMVN by utterance, default: False.")
    parser.add_argument("--speed-perturbation", type=float, dest='sp',
                        nargs='*', default=[], help=f"Add speed perturbation to subset(s).")
    args = parser.parse_args()

    assert os.path.isdir(args.src_data)
    subsets_ = args.subset

    for _sp_factor in args.sp:
        assert (isinstance(_sp_factor, float) or isinstance(_sp_factor, int)) and _sp_factor > 0, \
            f"Unsupport speed pertubation value: {_sp_factor}"

    d_sets = [os.path.join(args.src_data, _set) for _set in subsets_]

    from cat.utils.data import data_prep
    data_prep.prepare_kaldi_feat(
        subsets=subsets_,
        trans=[f"{path}/text" for path in d_sets],
        audios=[f"{path}/wav.scp" for path in d_sets],
        num_mel_bins=80,
        apply_cmvn=args.cmvn,
        speed_perturb=args.sp,
        read_from_extracted_meta=True
    )
