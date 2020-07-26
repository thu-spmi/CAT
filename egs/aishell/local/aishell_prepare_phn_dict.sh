#!/bin/bash

# This script prepares the phoneme-based lexicon. It also generates the list of lexicon units
# and represents the lexicon using the indices of the units. 

srcdir=data/local/train
dir=data/local/dict_phn
mkdir -p $dir
srcdict=data/resource_aishell/lexicon.txt

[ -f path.sh ] && . ./path.sh

[ ! -f "$srcdict" ] && echo "No such file $srcdict" && exit 1;

# Raw dictionary preparation
cat $srcdict | grep -v "!SIL" | \
  perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; print $_;}}' \
  > $dir/lexicon_raw.txt || exit 1;

# Get the set of lexicon units without noises
cut -d' ' -f2- $dir/lexicon_raw.txt | tr ' ' '\n' | sort -u   > $dir/units_raw.txt

(echo '<SPOKEN_NOISE> <SPN>'; echo '<UNK> <SPN>'; echo '<NOISE> <NSN>'; ) | \
 cat - $dir/lexicon_raw.txt | sort | uniq > $dir/lexicon.txt || exit 1;

(echo '<NSN>'; echo '<SPN>';) | cat - $dir/units_raw.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Phoneme-based dictionary preparation succeeded"
