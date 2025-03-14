#!/bin/bash
# Copyright 2025 Tsinghua SPMI Lab
# Author: Sardar (sar_dar@foxmail.com)

set -e -u
<<"PARSER"
("dir", type=str,help="SPG-JSA exp dir")
("lang_dir", type=str, help="the path of lang dir that include lexicon, lm dir and WFST graph dir.")
("--sta", type=int, default=0, help="Start stage. Default: 0")
("--sto", type=int, default=4, help="Stop stage. Default: 4")
("--lw_range", type=str, default='0.1,1.0', help="the range of LM weight. Default: 0.1~1.0")
("--lw", type=str, default='None', help="the LM weight for WFST based n-best list producing. Default: None")
PARSER
eval "$(python utils/parseopt.py $0 $*)"

function get_tokenizer() {
    echo $(
        python -c \
            "import json;print(json.load(open('$1/hyper-p.json'))['tokenizer']['file'])"
    )
}

graph_dir=$lang_dir/graph
lm_dir=$lang_dir/lm
word_list=$lm_dir/word_list.txt
echo "### exp dir: $dir"

if [ ${sta} -le 1 ] && [ ${sto} -ge 1 ]; then
    echo "### stage 1: preparing decode LM with arpa format ..."
    python local/word_list.py --dir $lm_dir --out $word_list --dump2json
    lm="$lm_dir/4gram.arpa"
    bash utils/pipeline/ngram.sh $lm_dir \
        -o 4 --arpa --output $lm --stop-stage 3
    echo "### Decode LM saved at $lm"
fi

if [ ${sta} -le 2 ] && [ ${sto} -ge 2 ]; then
    echo "### stage 2: building decoding graph TLG.fst ..."
    bash utils/tool/build_decoding_graph.sh --word_list $word_list \
        $(get_tokenizer $dir) \
        $(get_tokenizer $lm_dir ) \
        $lm_dir/4gram.arpa $graph_dir
    echo "### building TLG.fst is finished!"
fi

if [ ${sta} -le 3 ] && [ ${sto} -ge 3 ]; then
    echo "### stage 3: FST decoding ..."
    start=$(echo $lw_range | cut -d',' -f1)
    end=$(echo $lw_range | cut -d',' -f2)
    if ! [[ $start =~ ^[0-9]+(\.[0-9]+)?$ ]] || ! [[ $end =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "${lw_range} not a valid float number range."
        exit 1
    fi
    current_lw=$start
    echo "### decoding graph: ${graph_dir}"
    while (( $(echo "$current_lw <= $end" | bc) )); do
        bash local/eval_fst_decode.sh \
            $dir \
            $graph_dir \
            --lmwt $current_lw \
            --mode wer 
        current_lw=$(echo "$current_lw + 0.1" | bc)
    done
    echo "### FST decoding is finished!"
fi

if [ ${sta} -le 4 ] && [ ${sto} -ge 4 ]; then
    echo "### stage 4: generating n-best file for MLS decoding ..."
    if [[ $lw == "None" ]]; then
        echo "WARNING: You seem to forget to set the --lw parameter.."
        $0 -h
        exit 1
    fi
    if ! [[ $lw =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "lw=${lw} is not a valid float number."
        exit 1
    fi
    echo "### decoding graph: ${graph_dir}"
    bash local/eval_fst_decode.sh \
        $dir \
        ${graph_dir} \
        --lmwt $lw \
        --mode wer \
        -f \
        --mls \
        --beam 32 --lattice-beam 32 \
        --nbest 64
    echo "### Generating the n-best file is finished!"
fi