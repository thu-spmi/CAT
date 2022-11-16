#!/bin/bash
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# This script prepare libri data by torchaudio
set -e -u

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)
set -e -u
<<"PARSER"
("-src", type=str, default="/data/LibriSpeech",
    help="Source data folder containing the audios and transcripts. "
        "Download from https://www.openslr.org/12")
("-sp", type=float, nargs='*', default=None,
    help="Speed perturbation factor(s). Default: None.")
("-subsets-fbank", type=str, nargs="+",
    choices=['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500'],
    default=['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500'],
    help="Subset(s) for extracting FBanks. Default all.")
PARSER
eval $(python utils/parseopt.py $0 $*)

opt_sp="1.0"
[ "$sp" != "None" ] && export opt_sp=$sp

python local/extract_meta.py $src \
    --subset $subsets_fbank --speed-perturbation $opt_sp || exit 1

python utils/data/resolvedata.py

echo "$0 done."
exit 0
