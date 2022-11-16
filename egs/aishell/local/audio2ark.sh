#!/bin/bash
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# This script directly save raw audios into kaldi ark (for experiment reading raw audios)
# Run local/data_kaldi.sh to obtain wav.scp and text before this script.
set -e -u

export KALDI_ROOT="/opt/kaldi"
datasets="
dev
test
train
"

src=$(readlink -f data/src)
cd $KALDI_ROOT/egs/wsj/s5 && . ./path.sh
cd - >/dev/null

for set in $datasets; do
    d_data=$src/${set}_raw
    mkdir -p $d_data
    cp $src/$set/text $d_data
    [[ -f $d_data/feats.scp && $(wc -l $d_data/feats.scp | awk '{print $1}') -eq $(wc -l $d_data/text | awk '{print $1}') ]] &&
        continue

    python local/wav_norm.py $src/$set/wav.scp \
        ark,scp:$d_data/audio.ark,$d_data/feats.scp
done

python utils/data/resolvedata.py
echo "$0 done."
exit 0
