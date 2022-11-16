#!/bin/bash
# Author: Huahuan Zheng (maxwellzh@outlook.com)
set -e -u

dir="exp/asr-ctc-lexicon"
KALDI_ROOT=/opt/kaldi
export KALDI_ROOT=$KALDI_ROOT

# prepare a word to char lexicon
mkdir -p $dir/local
[ ! -f $dir/local/lexicon.txt ] &&
    for w in YES NO; do
        echo "$w $(echo $w | grep -o . | xargs)" \
            >>$dir/local/lexicon.txt
    done

# model training
python utils/pipeline/asr.py $dir

# prepare decoding graph
## train a word based LM
lm="$dir/decode_lm/2gram.arpa"
bash utils/pipeline/ngram.sh $dir/decode_lm \
    -o 2 --arpa --output $lm --sto 3

function get_tokenizer() {
    echo $(
        python -c \
            "import json;print(json.load(open('$1/hyper-p.json'))['tokenizer']['file'])"
    )
}
# prepare decoding graph: will create TLG.fst and r_words.txt in $dir/graph
bash utils/tool/build_decoding_graph.sh \
    $(get_tokenizer $dir) \
    $(get_tokenizer $dir/decode_lm) \
    $lm $dir/graph

# decoding and eval
bash local/eval_fst_decode.sh \
    $dir $dir/graph \
    --data yesno

echo "$0 done."
exit 0
