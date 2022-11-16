#!/bin/bash
# Author: Huahuan Zheng
set -e -u
<<"PARSER"
("dir", type=str, nargs='?', default=$(dirname $0),
    help="Input file.")
("-o", "--output", type=str, default="${input}_out",
    help="Output file. Default: <input>_out")
PARSER
eval $(python utils/parseopt.py $0 $*)

KALDI_ROOT="/opt/kaldi"
export KALDI_ROOT=$KALDI_ROOT
function get_tokenizer() {
    echo $(
        python -c \
            "import json;print(json.load(open('$1/hyper-p.json'))['tokenizer']['file'])"
    )
}

# prepare data and lexicon
bash local/data_kaldi.sh -use-3way-sp

# prepare the tokenizer with stage 1
python utils/pipeline/asr.py $dir --sto 1

# prepare den_lm.fst
[ ! -f $dir/den_lm.fst ] && {
    ftrans="data/src/train_si284/text"
    [ ! -f $ftrans ] && {
        echo "Default transcript $ftrans not found."
        echo "Please update it."
        exit 1
    }
    bash utils/tool/prep_den_lm.sh \
        -tokenizer=$(get_tokenizer $dir) \
        -kaldi-root=$KALDI_ROOT \
        $ftrans $dir/den_lm.fst
    echo "Denominator LM saved at $dir/den_lm.fst"
}

# finish rest of the stages
python utils/pipeline/asr.py $dir --sta 2

# prepare decoding graph
lm="$dir/decode_lm/4gram.arpa"
[ ! -f $lm ] && {
    bash utils/pipeline/ngram.sh $dir/decode_lm \
        -o 4 --arpa --output $lm --sto 3
    echo "Decode LM saved at $lm"
}

# prepare decoding TLG.fst
bash utils/tool/build_decoding_graph.sh \
    $(get_tokenizer $dir) \
    $(get_tokenizer $dir/decode_lm) \
    $lm $dir/graph

echo "TLG.fst finsh"

# TLG decoding
bash ../TEMPLATE/local/eval_fst_decode.sh \
    $dir{,/graph} \
    --data test_dev93 test_eval92 \
    --acwt 1.0 --lmwt 1.0 -f

exit 0
