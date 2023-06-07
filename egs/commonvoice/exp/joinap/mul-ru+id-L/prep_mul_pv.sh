#!/bin/bash

lang="ru id"

dlang="data/lang-mul"
echo $lang >>$dlang/lang.txt
mkdir -p $dlang
export LC_ALL=C.UTF-8

for l in $lang; do
    cat data/lang-$l/lexicon
done | sort -k 1,1 -u -s \
    >$dlang/lexicon

cut <$dlang/lexicon -f 2- | tr ' ' '\n' | sort -u -s >$dlang/phonemes.txt

[ ! -f local/data/ipa_all.csv ] && {
    wget https://raw.githubusercontent.com/dmort27/panphon/master/panphon/data/ipa_all.csv \
        -O local/data/ipa_all.csv
}
python local/get_ipa_mapping.py \
    $dlang/phonemes.txt \
    local/data/ipa_all.csv \
    $dlang/mul-pv.npy || exit 1

echo "$0 done" && exit 0
