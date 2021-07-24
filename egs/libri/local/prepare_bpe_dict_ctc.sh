#!/bin/bash

# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Wenjie Peng (wenjayep@gmail.com)

# This script implements lexicon construction for wordpiece or char based systems.

. ./path.sh
. ./cmd.sh

echo $#

text=$1 # data/train_tr95/text
src=$2  # data/local/dict_phn
dir=$3  # data/local/dict_bpe
mkdir -p $dir
bpemode=char # `unigram` for wordpiece-based system
bpemodel=$dir/train_${bpemode}
nbpe=150
unk_id=2

cut -f 2- -d ' ' $text > $dir/input.txt

# train bpe model.
spm_train --input=$dir/input.txt --vocab_size=${nbpe} --bos_id=0 --eos_id=1 --unk_id=$unk_id \
    --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 \
    --treat_whitespace_as_suffix=false --unk_surface="<UNK>"

## encode text into bpe ids.
for x in train_tr95 train_cv05 dev_clean dev_other test_clean test_other; do
    cut -f 2- -d ' ' data/$x/text > data/$x/text.tmp
    spm_encode --model=${bpemodel}.model --output_format=id < data/${x}/text.tmp > data/${x}/text.id_tmp || exit 1;
    spm_encode --model=${bpemodel}.model --output_format=piece < data/${x}/text.tmp > data/${x}/text.piece_tmp || exit 1;
    awk '{print $1}' data/$x/text > data/$x/text.uttid
    paste -d " " data/$x/text.uttid data/$x/text.id_tmp > data/$x/text.id
    paste -d " " data/$x/text.uttid data/$x/text.piece_tmp > data/$x/text.piece
    rm data/$x/text.{id_tmp,piece_tmp}
done


awk '{print NR " " NR}' ${bpemodel}.vocab | grep -v '^0 0$' | grep -v '^1 1$' | awk '{print $1 " " NR}' > $dir/units.txt

awk '{print $1}' $src/lexicon.txt > $dir/lexicon.tmp

spm_encode --model=${bpemodel}.model --output_format=id  < $dir/lexicon.tmp | paste -d " " \
    $dir/lexicon.tmp - | grep -v '<' > $dir/lexicon.txt

echo "<NOISE> $unk_id" >> $dir/lexicon.txt
echo "<SPOKEN_NOISE> $unk_id" >> $dir/lexicon.txt
echo "<UNK> $unk_id" >> $dir/lexicon.txt

utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt
echo "Succeeded in generating bpe dict."
