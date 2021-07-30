#!/bin/bash

# Copyright Johns Hopkins University (Author: Daniel Povey) 2012
#           Guoguo Chen 2014

# This script is adopted from wsj of Kaldi, modified by Chengrui Zhu.

lang_suffix=
stage=1
echo "$0 $@"  # Print the command line for logging
. ./path.sh
. utils/parse_options.sh || exit 1;

[ ! -d data/lang_${lang_suffix} ] &&\
  echo "Expect data/lang_${lang_suffix} to exist" && exit 1;

lm_srcdir_3g=data/local/local_lm/3gram-mincount
lm_srcdir_4g=data/local/local_lm/4gram-mincount

[ ! -d "$lm_srcdir_3g" ] && echo "No such dir $lm_srcdir_3g" && exit 1;
[ ! -d "$lm_srcdir_4g" ] && echo "No such dir $lm_srcdir_4g" && exit 1;

for d in data/lang_${lang_suffix}_test_bd_{tg,tgpr,tgconst,fg,fgpr,fgconst}; do
  rm -r $d 2>/dev/null
  mkdir -p $d
  cp -r data/lang_${lang_suffix}/* $d
done

lang=data/lang_${lang_suffix}

# Parameters needed for ConstArpaLm.
unk=`grep "<UNK>" $lang/words.txt | awk '{print $2}'`
bos=`grep "<s>" $lang/words.txt | awk '{print $2}'`
eos=`grep "</s>" $lang/words.txt | awk '{print $2}'`
if [[ -z $bos || -z $eos ]]; then
  echo "$0: <s> and </s> symbols are not in $lang/words.txt"
  exit 1;
fi

tmpdir=data/lang_${lang_suffix}_tmp
mkdir -p $tmpdir

# Be careful: this time we dispense with the grep -v '<s> <s>' so this might
# not work for LMs generated from all toolkits.
if [ $stage -le 1 ]; then
echo "build 3-gram TLG.fst"
gunzip -c $lm_srcdir_3g/lm_pr6.0.gz | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$lang/words.txt - data/lang_${lang_suffix}_test_bd_tgpr/G.fst || exit 1;
  fstisstochastic data/lang_${lang_suffix}_test_bd_tgpr/G.fst
  fsttablecompose $lang/L.fst data/lang_${lang_suffix}_test_bd_tgpr/G.fst | fstdeterminizestar \
      --use-log=true | fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst \
      || exit 1;
  fsttablecompose $lang/T.fst $tmpdir/LG.fst > data/lang_${lang_suffix}_test_bd_tgpr/TLG.fst \
      || exit 1;

gunzip -c $lm_srcdir_3g/lm_unpruned.gz | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$lang/words.txt - data/lang_${lang_suffix}_test_bd_tg/G.fst || exit 1;
  fstisstochastic data/lang_${lang_suffix}_test_bd_tg/G.fst
  fsttablecompose $lang/L.fst data/lang_${lang_suffix}_test_bd_tg/G.fst | fstdeterminizestar \
      --use-log=true | fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst \
      || exit 1;
  fsttablecompose $lang/T.fst $tmpdir/LG.fst > data/lang_${lang_suffix}_test_bd_tg/TLG.fst \
      || exit 1;


echo "build 4-gram TLG.fst"
gunzip -c $lm_srcdir_4g/lm_unpruned.gz | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$lang/words.txt - data/lang_${lang_suffix}_test_bd_fg/G.fst || exit 1;
  fstisstochastic data/lang_${lang_suffix}_test_bd_fg/G.fst
  fsttablecompose $lang/L.fst data/lang_${lang_suffix}_test_bd_fg/G.fst | fstdeterminizestar \
      --use-log=true | fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst \
      || exit 1;
  fsttablecompose $lang/T.fst $tmpdir/LG.fst > data/lang_${lang_suffix}_test_bd_fg/TLG.fst \
      || exit 1;


gunzip -c $lm_srcdir_4g/lm_pr7.0.gz | \
  arpa2fst --disambig-symbol=#0 \
           --read-symbol-table=$lang/words.txt - data/lang_${lang_suffix}_test_bd_fgpr/G.fst || exit 1;
  fstisstochastic data/lang_${lang_suffix}_test_bd_fgpr/G.fst
  fsttablecompose $lang/L.fst data/lang_${lang_suffix}_test_bd_fgpr/G.fst | fstdeterminizestar \
      --use-log=true | fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst \
      || exit 1;
  fsttablecompose $lang/T.fst $tmpdir/LG.fst > data/lang_${lang_suffix}_test_bd_fgpr/TLG.fst \
      || exit 1;

fi
echo "Build ConstArpaLm for the unpruned language model."
# Build ConstArpaLm for the unpruned language model.
#TODO: 
gunzip -c $lm_srcdir_3g/lm_unpruned.gz | \
  utils/map_arpa_lm.pl $lang/words.txt | \
  arpa-to-const-arpa --bos-symbol=$bos --eos-symbol=$eos --unk-symbol=$unk \
  - data/lang_${lang_suffix}_test_bd_tgconst/G.carpa || exit 1

gunzip -c $lm_srcdir_4g/lm_unpruned.gz | \
  utils/map_arpa_lm.pl $lang/words.txt | \
  arpa-to-const-arpa --bos-symbol=$bos --eos-symbol=$eos --unk-symbol=$unk \
  - data/lang_${lang_suffix}_test_bd_fgconst/G.carpa || exit 1

exit 0

