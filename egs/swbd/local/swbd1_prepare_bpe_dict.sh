#!/bin/bash

# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Wenjie Peng (wenjayep@gmail.com)

# This script implements lexicon construction for wordpiece or char based systems.


. ./path.sh
. ./cmd.sh


srcdir=data/local/train_nohup_sp
dir=data/local/dict_bpe
nbpe=150 
bpemode=unigram
bpemodel=$dir/train_${bpemode}
unk_id=2
mkdir -p $dir
#srcdict=$srcdir/swb_ms98_transcriptions/sw-ms98-dict.text

cat data/train_nodup/text | cut -f 2- -d ' ' | tr 'A-Z' 'a-z' >  $dir/train.txt
sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' $dir/train.txt

# train bpe model
spm_train --input=$dir/train.txt --vocab_size=${nbpe} --bos_id=0 \
    --eos_id=1 --unk_id=$unk_id --model_type=${bpemode} --model_prefix=${bpemodel} \
    --input_sentence_size=100000000 --user_defined_symbols="[laughter],[noise]"\
    --unk_piece="[vocalized-noise]" --treat_whitespace_as_suffix=false --unk_surface="<unk>"

## convert text into bpe id
for x in train_nodup train_dev; do
    cp data/$x/text data/$x/text.pos
    sed -i 's/\._/ /g; s/\.//g; s/them_1/them/g' data/$x/text.pos
    cut -f 2- -d ' ' data/$x/text.pos | tr 'A-Z' 'a-z' > data/$x/text.tmp
    awk '{print $1}' data/$x/text.pos > data/$x/text.uttid
    spm_encode --model=${bpemodel}.model --output_format=id < data/${x}/text.tmp > data/${x}/text.id_tmp || exit 1;
    spm_encode --model=${bpemodel}.model --output_format=piece < data/${x}/text.tmp > data/${x}/text.piece_tmp || exit 1;
    paste -d ' ' data/$x/text.uttid data/$x/text.id_tmp > data/$x/text.id
    paste -d ' ' data/$x/text.uttid data/$x/text.piece_tmp > data/$x/text.piece
done

# get units.txt
cat data/train_nodup/text.id | cut -f 2- -d ' ' | tr ' ' '\n' | tr 'A-Z' 'a-z' | sort -n \
    | uniq | grep -v '^0$' | grep -v '^1$' | awk '{print $1 " " NR}' > $dir/units.txt

# get lexicon.txt
tr ' ' '\n' < $dir/train.txt | grep -v '^$' | sort | uniq | grep -v '\[vocalized-noise\]' \
    > $dir/train.wrd
spm_encode --model=${bpemodel}.model --output_format=id  < $dir/train.wrd | paste -d ' ' \
    $dir/train.wrd - > $dir/lexicon.txt || exit 1;

echo "<unk> $unk_id" >> $dir/lexicon.txt
echo "[vocalized-noise] $unk_id" >> $dir/lexicon.txt

grep -v '\[laughter|noise|vocalized-noise\]' $dir/lexicon.txt > $dir/lexicon_raw_nosil.txt
grep -v '\[laughter|noise|vocalized-noise\]' $dir/units.txt > $dir/units_nosil.txt

utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Succeed in generating dict"
