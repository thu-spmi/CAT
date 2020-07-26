#!/bin/bash

# Copyright 2014 Vassil Panayotov
# Apache 2.0

# Prepares the dictionary and auto-generates the pronunciations for the words,
# that are in our vocabulary but not in CMUdict

stage=0

. utils/parse_options.sh || exit 1;
. ./path.sh || exit 1

lm_dir=$1
dir=$2

vocab=$lm_dir/librispeech-lexicon.txt
[ ! -f $vocab ] && echo "$0: vocabulary file not found at $vocab" && exit 1;

mkdir -p $dir || exit 1;
echo $dir

cat $vocab | \
  perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; $s = join(" ",@A); print $s; print "\n"}}' \
  > $dir/lexicon_raw_nosil.txt || exit 1;
echo "lexicon_raw_nosil done"

# awk '{for (i=2; i<=NF; ++i) { print $i; gsub(/[0-9]/, "", $i); print $i}}' $dir/lexicon_raw_nosil.txt |\
#   sort -u |\
#   perl -e 'while(<>){
#     chop; m:^([^\d]+)(\d*)$: || die "Bad phone $_";
#     $phones_of{$1} .= "$_ "; }
#     foreach $list (values %phones_of) {print $list . "\n"; } ' | sort \
#     > $dir/units_nosil.txt || exit 1;

cut -d' ' -f2- $dir/lexicon_raw_nosil.txt | tr ' ' '\n' | sort -u > $dir/units_nosil.txt
echo "units_nosil.txt done"

(echo '<SPOKEN_NOISE> <SPN>'; echo '<UNK> <SPN>'; echo '<NOISE> <NSN>'; ) | \
 cat - $dir/lexicon_raw_nosil.txt | sort | uniq > $dir/lexicon.txt || exit 1;

#  The complete set of lexicon units, indexed by numbers starting from 1
(echo '<NSN>'; echo '<SPN>';) | cat - $dir/units_nosil.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert character sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt
echo "lexicon_numbers.txt done"
