#!/bin/bash
# author: Huahuan Zheng
# This script shows how to train a CTC-CRF model.
set -e

KALDI_ROOT=/opt/kaldi
DIR=$(dirname $0)

[ -z $KALDI_ROOT ] && {
    echo "\$KALDI_ROOT is not set."
    exit 1
}

# prepare data
bash local/data.sh

# prepare tokenizer
python utils/pipeline/asr.py $DIR --sto 1

# prepare den lm
[ ! -f $DIR/den_lm.fst ] &&
    {
        f_text=$(python -c "
import sys;
sys.path.append('.')
import utils.pipeline.common_utils as cu
files = ' '.join(sum(cu.find_text(cu.readjson('$DIR/hyper-p.json')['data']['train']), []))
print(files)")
        for x in $f_text; do
            [ ! -f $x ] && echo "No such training corpus: '$x'" && exit 1
        done
        cat $f_text | bash utils/tool/prep_den_lm.sh \
            -tokenizer="$DIR/tokenizer.tknz" \
            -kaldi-root=$KALDI_ROOT \
            -ngram-order=3 \
            -no-prune-ngram-order=2 \
            /dev/stdin $DIR/den_lm.fst
    }

echo "Denominator LM stored at $DIR/den_lm.fst"

# prepare decode lm
bash utils/pipeline/ngram.sh $DIR/decode-lm -o 3 --sto 3

# finish rest stages
python utils/pipeline/asr.py $DIR --sta 2 --ngpu 1

echo "$0 done."
exit 0
