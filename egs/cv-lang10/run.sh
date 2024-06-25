#!/bin/bash

# Copyright 2024 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# Acknowlegement: This script refer to the code of Huahuan Zheng (maxwellzh@outlook.com)
# This script includes the process of data preparation, model training and decoding.

set -e -u
<<"PARSER"
("lang", type=str, nargs='?', default='en', help="language name.")
("dir", type=str, nargs='?', default='exp',
    help="Input file.")
("--mode", type=str, default='phone', help="Type of model, phone or subword. Default: phone")
("--dict_dir", type=str, default='dict', help="dir of lexicon or word list")
("--sta", type=int, default=0,
    help="Start stage. Default: 0")
("--sto", type=int, default=7,
    help="Stop stage. Default: 7")

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
function get_train_tran() {
    echo $(
        python -c \
            "import json;print(json.load(open('data/metainfo.json'))['$1']['trans'])"
    )
}
if [ ${lang} -le 'ten' ];then
    langs="en es nl fr it ru ky sv-SE tr tt"
else
    langs=$lang
fi

stage=$sta
stop_stage=$sto
dict_dir=dict  # lang dir including lexicon and word list


echo "### exp dir: $dir"
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "### stage 0: preparing data..."
    for lg in langs; do
        bash local/data_prep.sh $lg
        echo "$lg data is done"
    done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "### stage 1: preparing tokenizer ..."
    python utils/pipeline/asr.py $dir --sta 1 --sto 1
    echo "### tokenizer is done!"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "### stage 2: packing data ..."
    python utils/pipeline/asr.py $dir --sta 2 --sto 2
    echo "### packing data is done"
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "### stage 3: training model ..."
    python utils/pipeline/asr.py $dir --sta 3 --sto 3
    echo "### training model is done"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "### stage 4: CTC decoding ..."
    python utils/pipeline/asr.py $dir --sta 4
    echo "### Decoding is done"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    for lg in $langs; do
        echo "### stage 5: preparing decode $lg LM with arpa format ..."
        lm_dir="$dict_dir/$lg/lm"
        lm="$lm_dir/4gram.arpa"
        bash utils/pipeline/ngram.sh $lm_dir \
            -o 4 --arpa --output $lm --stop-stage 3
        echo "### Decode LM saved at $lm"
    done
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    for lg in $langs; do
        echo "### stage 6: building $lg decoding graph TLG.fst ..."
        if [ $mode == "subword" ];then
          awk '{print $1}' $dict_dir/$lg/lexicon.txt > $dict_dir/$lg/word_list
          word_list=$dict_dir/$lg/word_list
          bash utils/tool/build_decoding_graph.sh --word_list $word_list \
            $(get_tokenizer $dir) \
            $(get_tokenizer $dict_dir/$lg/lm) \
            $dict_dir/$lg/lm/4gram.arpa $dict_dir/$lg/graph_subword
        else
          bash utils/tool/build_decoding_graph.sh \
            $(get_tokenizer $dir) \
            $(get_tokenizer $dict_dir/$lg/lm) \
            $dict_dir/$lg/lm/4gram.arpa $dict_dir/$lg/graph_phn
        fi
        echo "### $lg TLG.fst finish"
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    for lg in $langs; do
        echo "### stage 7: $lg FST decoding ..."
        score=(0.8 0.9 1.0)
        if [ $mode == "subword" ];then
            graph_dir=$dict_dir/$lg/graph_subword
        else
            graph_dir=$dict_dir/$lg/graph_phn
        fi
        for lm in ${score[@]}; do
            bash local/eval_fst_decode.sh \
                $dir \
                $graph_dir \
                --data test_${lg} \
                --lmwt $lm \
                --mode wer -f
        done
        echo "### $lg fst decode finshed!"
    done
fi
exit 0
