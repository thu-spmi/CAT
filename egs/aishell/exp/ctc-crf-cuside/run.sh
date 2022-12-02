#!/bin/bash
set -e -u 
<<"PARSER"
("dir",type=str,nargs='?',default=$(dirname $0),
    help="Input file.")
("-o","--output",type=str,default="${input}_out",
    help="Output file.Default:<input>_out")
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

# prepare lexion and userdict
bash exp/ctc-crf-cuside/run_lexicon.sh

# prepare the tokenizer with stage 1
python utils/pipeline/asr.py $dir --sto 1

# prepare den_lm.fst
[ ! -f $dir/den_lm.fst ] && {
    trans="data/src/train/text"
    [ ! -f $trans ] && {
        echo "Default transcript $trans not found."
        echo "Please update it."
        exit 1
    }
    bash utils/tool/prep_den_lm.sh \
        -tokenizer=$(get_tokenizer $dir) \
        -kaldi-root=$KALDI_ROOT \
        $trans $dir/den_lm.fst
    echo "Denominator LM save"
}

# prepare decoding graph
lm="$dir/decode_lm/3gram.arpa"
[ ! -f $lm ] && {
    bash utils/pipeline/ngram.sh $dir/decode_lm \
        -o 3 --arpa --output $lm --sto 3
    echo "Decode LM saved at $lm"
}

# prepare decoding TLG.fst
bash utils/tool/build_decoding_graph.sh \
    $(get_tokenizer $dir) \
    $(get_tokenizer $dir/decode_lm) \
    $lm $dir/graph

echo "TLG.fst finsh"

# finsh asr training
python utils/pipeline/asr.py $dir --sta 2

# TLG decoding
bash ../TEMPLATE/local/eval_fst_decode.sh \
    $dir/graph \
    --data test \
    --acwt 1.0 --lmwt 1.1 -f

exit 0
