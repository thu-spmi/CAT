#!/bin/bash

# This script prepares the phoneme-based lexicon. It also generates the list of lexicon units
# and represents the lexicon using the indices of the units. 

dir=${H}/data/dict
mkdir -p $dir
srcdict=input/lexicon.txt

. ./path.sh

# Check if lexicon dictionary exists
[ ! -f "$srcdict" ] && echo "No such file $srcdict" && exit 1;

# Raw dictionary preparation
# grep removes SIL, perl removes repeated lexicons
cat $srcdict | grep -v "SIL" | \
  perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; print $_;}}' \
  > $dir/lexicon_raw.txt || exit 1;

# Get the set of units in the lexicon without noises
# cut: remove words, tr: remove spaces and lines, sort -u: sort and unique
cut -d ' ' -f 2- $dir/lexicon_raw.txt | tr ' ' '\n' | sort -u > $dir/units_raw.txt

# add noises for lexicons
(echo '<SPOKEN_NOISE> <SPN>'; echo '<UNK> <SPN>'; echo '<NOISE> <NSN>'; ) | \
 cat - $dir/lexicon_raw.txt | sort | uniq > $dir/lexicon.txt || exit 1;

# add noises and number the units
(echo '<NSN>'; echo '<SPN>';) | \
 cat - $dir/units_raw.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Phoneme-based dictionary preparation succeeded"
