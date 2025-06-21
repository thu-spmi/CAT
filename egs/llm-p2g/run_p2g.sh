#!/bin/bash

# Copyright 2025 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# Acknowlegement: This script refer to the code of Huahuan Zheng (maxwellzh@outlook.com)
# This script includes the process of data preparation, model training and decoding.

set -e -u
<<"PARSER"
("lang", type=str, nargs='?', default='pl', help="language name.")
("dir", type=str, nargs='?', default='exp',
    help="Input file.")
("--mode", type=str, default='danp', help="Type of model, danp or tkm. Default: danp")
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

lang="pl"

stage=$sta
stop_stage=$sto
dict_dir=dict  # lang dir including lexicon and word list
data_dir=data  # data dir including text



echo "### exp dir: $dir"
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "### stage 0: preparing data..."
    bash local/data_prep.sh $lang
    echo "$lang data is done"
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "### stage 1: preparing tokenizer ..."
    if [ $mode == "danp" ];then
        python utils/pipeline/p2g.py $dir --sta 1 --sto 1
    else
        python utils/pipeline/p2g_tkm.py $dir --sta 1 --sto 1
    echo "### tokenizer is done!"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "### stage 2: packing data ..."
    if [ $mode == "danp" ];then
        python utils/pipeline/p2g.py $dir --sta 2 --sto 2
    else
        python utils/pipeline/p2g_tkm.py $dir --sta 2 --sto 2
    echo "### packing data is done"
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "### stage 3: training model ..."
    if [ $mode == "danp" ];then
        python utils/pipeline/p2g.py $dir --sta 3 --sto 3
    else
        python utils/pipeline/p2g_tkm.py $dir --sta 3 --sto 3
    echo "### training model is done"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "### stage 4: AED decoding ..."
    if [ $mode == "danp" ];then
        python utils/pipeline/p2g.py $dir --sta 4
    else
        python utils/pipeline/p2g_tkm.py $dir --sta 4
    echo "### AED Decoding is done"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "### stage 5: Rescoring with $lang LM ..."
    gt_text=$data_dir/$lang/test/text
    tokenizer=$dict_dir/$lang/lm/tokenizer_lm.tknz
    config=$dict_dir/$lang/lm/config.json
    alpha=(0.01 0.02 0.03 0.04 0.05)  # For TKM, it should be (0.1 0.2 0.3 0.4 0.5)
    for lm in ${alpha[@]}; do
        hy_text=${dir}/decode/rescoing_lm_${lm}
        python ../../cat/lm/rescore.py $nbest_text $hy_text \
        --alpha $lm --tokenizer $tokenizer --save-lm-nbest ${out_hy_text}_${lm}.nbest --config $config
    done 
    for lm in ${alpha[@]}; do
        echo lm_${lm}
        python ../../cat/utils/wer.py $gt_text ${out_hy_text}_${lm}
    done
    echo "### Rescoring is done"
fi
exit 0
