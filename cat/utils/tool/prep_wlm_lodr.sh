#!/bin/bash
# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# 
# Train the pruned bigram LM described in LODR method:
# "An Empirical Study of Language Model Integration for Transducer based Speech Recognition"
# https://arxiv.org/abs/2203.16776

set -e
set -u

<<"PARSER"
("dir", type=str, help="Path to the LM directory.")
("--topk", type=int, default=20000, help="Top-k of 2-grams. default: 20000")
("-o", "--order", type=int, default=2, help="Max order of n-gram. default: 2")
PARSER
eval $(python utils/parseopt.py $0 $*)

export PATH=../../src/bin:$PATH
[ ! $(command -v count_ngrams) ] && (
    echo "KenLM not installed? Command not found: count_ngrams"
    exit 1
)

[ ! $(command -v python) ] && (
    echo "no python executable in PATH."
    exit 1
)

[ ! -f utils/pipeline/lm.py ] && (
    echo "no utils/pipeline/lm.py found."
    exit 1
)

# preprare the data.
python utils/pipeline/lm.py --sto 2 $dir

export corpus_count="/tmp/$(
    head /dev/urandom | tr -dc A-Za-z0-9 | head -c10
).corpus.tmp"
export vocab="/tmp/$(
    head /dev/urandom | tr -dc A-Za-z0-9 | head -c10
).corpus.tmp"
python utils/data/corpus2index.py $dir/pkl/train.pkl --map 0: 1: |
    count_ngrams -o $order -S 20% --write_vocab_list $vocab \
        >$corpus_count
read -r prunearg realtopk <<<$(python utils/tool/get_prune_args.py $corpus_count $order $topk)
rm -f $corpus_count $vocab

[ $realtopk != $topk ] && (
    echo "Query for top: $topk, but finally set top: $realtopk."
)

prune=$(python -c "print('0 '*($order-1),end='')")
prune="$prune $prunearg"

bash utils/pipeline/ngram.sh $dir \
    --start 3 -o $order \
    --prune $prune
