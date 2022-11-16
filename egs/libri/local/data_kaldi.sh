#!/bin/bash
# author: Huahuan Zheng
# this script prepare the libri data by kaldi tool
set -e

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

<<"PARSER"
("-src", type=str, default="/data/Librispeech",
    help="Source data folder containing the audios and transcripts. "
        "Download from https://www.openslr.org/12")
("-use-3way-sp", action="store_true",
    help="Use 3-way speed perturbation.")
("-subsets-fbank", type=str, nargs="+",
    choices=['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500'],
    default=['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500'],
    help="Subset(s) for extracting FBanks. Default all.")
PARSER
eval $(python utils/parseopt.py $0 $*)

[ -z $KALDI_ROOT ] && {
    echo "\$KALDI_ROOT is not set. re-run with"
    echo "KALDI_ROOT=xxx $0 $*"
    exit 1
}
export KALDI_ROOT=$KALDI_ROOT

if [ $use_3way_sp == "True" ]; then
    opt_3way_sp="--apply-3way-speed-perturbation"
else
    opt_3way_sp=""
fi

# extract meta
python local/extract_meta_kaldi.py $src \
    --subset $subsets_fbank

# compute fbank feat
# by default, we use 80-dim raw fbank and do not apply
# ... the CMVN, which matches the local/data.sh via torchaudio
bash utils/data/data_prep_kaldi.sh \
    $(printf "data/src/%s " $subsets_fbank) \
    --feat-dir=data/fbank \
    --nj=48 \
    --not-apply-cmvn \
    $opt_3way_sp

# refresh the data/metainfo.json file
python utils/data/resolvedata.py

echo "$0 done."
exit 0
