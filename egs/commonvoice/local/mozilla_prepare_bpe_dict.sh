#!/bin/bash

# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Wenjie Peng (wenjayep@gmail.com)

# This script implements lexicon construction for wordpiece or char based systems.
. ./cmd.sh
. ./path.sh

# punctuations removed

nbpe=150
bpemode=unigram # set `bpe|unigram` for wp-system, `char` for char-system
dir=data/local/dict_bpe
mkdir -p $dir
bpemodel=$dir/train_bpe${nbpe}

text=data/train_de/text
unk_id=2

# remove punctuations
set -euo pipefail

# use all the text data to generate bpe units

# remove puncuations of training text
cat data/train_de/text | cut -f 2- -d " " - | sed 's/"//g' | sed 's/,//g' | sed 's/\.//g' | sed 's/\?//g' | \
    sed 's/\!//g' | sed 's/…//g' | sed 's/;//g' | sed 's/  / /g' | sed 's/  / /g' | sed 's/ $//g' | sed "s/’/'/g" > $dir/input.txt
#
# train sentencepiece model using training text
# input: input text for training
# vocab_size: wp size. here $nbpe=150
# bos_id: id of <s>
# eos_id: id of </s>
# unk_id: id of <UNK> here $unk_id=2
# model_type: model for training spm_train, supported: [unigram, bpe, char, word]. here $bpemode=unigram
# model_prefix: prefix of the filename to save model 
# input_sentence_size: 
# unk_surface: symbol to replace unk_id when using spm_decode with wp id as input
python3 ctc-crf/spm_train --input=$dir/input.txt --vocab_size=${nbpe} --bos_id=0 --eos_id=1 --unk_id=$unk_id \
    --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 \
    --treat_whitespace_as_suffix=false --unk_surface="<UNK>"
#
#
for x in train dev test; do
    mkdir -p data/$x
    cp -r data/${x}_de/* data/${x}/
    cat  data/${x}/text | cut -f 2- -d " " - | sed 's/"//g' | sed 's/,//g' | sed 's/\.//g' | sed 's/\?//g' | sed 's/\!//g' \
        | sed 's/…//g' | sed 's/;//g' | sed 's/  / /g' | sed 's/  / /g' | sed 's/ $//g' | sed "s/’/'/g" > data/${x}/text.tmp
    awk '{print $1}' data/${x}/text > data/${x}/text.uttid
    paste -d " " data/${x}/text.uttid data/$x/text.tmp > data/$x/text_pos
    # model: path for saved model
    # output_format: specify encoded text format, support: [id, piece]
    python3 ctc-crf/spm_encode --model=${bpemodel}.model --output_format=id < data/${x}/text.tmp > data/${x}/text.id_tmp || exit 1;
    python3 ctc-crf/spm_encode --model=${bpemodel}.model --output_format=piece < data/${x}/text.tmp > data/${x}/text.piece_tmp || exit 1;
    paste -d ' ' data/${x}/text.uttid data/${x}/text.id_tmp > data/${x}/text.id
    paste -d ' ' data/${x}/text.uttid data/${x}/text.piece_tmp > data/${x}/text.piece
    rm data/$x/text.{id_tmp,piece_tmp}
done


# <s> and </s> needs to be removed for units.txt. Note normally text.id should not contain <s> and </s>, here we explicitly handle this.
cat  data/train/text.id | cut -f 2- -d " " | tr ' ' '\n' | sort -n | uniq | awk '{print $1 " " NR}' | grep -v "^0 0$" | grep -v "^1 1$" > $dir/units.txt



cat data/train/text.id | cut -f 2- -d " " - > $dir/train.tmp 
python3 ctc-crf/spm_decode --model=${bpemodel}.model --input_format=id < $dir/train.tmp | tr ' ' '\n' | sort | uniq | grep -v "^$" | grep -v '\<UNK\>' > $dir/train.wrd
#
python3 ctc-crf/spm_encode --model=${bpemodel}.model --output_format=id  < $dir/train.wrd | paste -d " " \
    $dir/train.wrd - > $dir/lexicon.txt || exit 1;

echo "<UNK> $unk_id" >> $dir/lexicon.txt
grep -v "<UNK>"  $dir/lexicon.txt > $dir/lexicon_raw_nosil.txt
grep -v "<UNK>" $dir/units.txt > $dir/units_nosil.txt

utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Succeed in generating dict"
