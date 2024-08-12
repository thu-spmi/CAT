#!/bin/bash

# Copyright 2024 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# Acknowlegement: This script refer to the code of Huahuan Zheng (maxwellzh@outlook.com)
# This script includes the process of data preparation, model training and decoding.

set -e -u
# <<"PARSER"
# ("langs", type=str, nargs='?', default='en', help="language name.")
# ("dir", type=str, nargs='?', default=$(dirname $0),
#     help="Input file.")
# ("--sta", type=int, default=0,
#     help="Start stage. Default: 0")
# ("--sto", type=int, default=7,
#     help="Stop stage. Default: 7")
# PARSER
# eval $(python utils/parseopt.py $0 $*)

# KALDI_ROOT="/opt/kaldi"
# export KALDI_ROOT=$KALDI_ROOT
function get_tokenizer() {
    echo $(
        python -c \
            "import json;print(json.load(open('$1/hyper-p.json'))['tokenizer']['file'])"
    )
}
function get_train_tran() {
    echo $(
        python -c \
            "import json;print(json.load(open('data/metainfo.json'))['$1']['trans'])"
    )
}

dir=exp/XLR-10-bpe
lm_dir=exp/decode_lm/
word_list=./dict/word_list
bash utils/tool/build_decoding_graph.sh --word_list $word_list \
            $(get_tokenizer $dir) \
            $(get_tokenizer $lm_dir) \
            $lm_dir/4gram.arpa $dir/graph_bpe
echo "### TLG.fst finish"

# echo "### stage 7: FST decoding ..."
score=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)
graph_dir=$dir/graph_bpe
# for lm in ${score[@]}; do
#     bash local/eval_fst_decode.sh \
#         $dir \
#         $graph_dir \
#         --data test_raw \
#                 --acwt 1.0 --lmwt $lm \
#         --mode wer -f
# done

bash local/eval_fst_decode.sh \
            $dir \
            $graph_dir \
            --data dev_raw \
                    --acwt 1.0 --lmwt 0.6 \
            --mode wer -f
echo "### fst decode finshed!"
