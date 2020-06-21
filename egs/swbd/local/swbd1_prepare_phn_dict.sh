#!/bin/bash

# This script prepares the phoneme-based lexicon. It also generates the list of lexicon units
# and represents the lexicon using the indices of the units. 

srcdir=data/local/train
dir=data/local/dict_phn
mkdir -p $dir
srcdict=$srcdir/swb_ms98_transcriptions/sw-ms98-dict.text

[ -f path.sh ] && . ./path.sh
. utils/parse_options.sh
[ ! -f "$srcdict" ] && echo "No such file $srcdict" && exit 1;

cp $srcdict $dir/lexicon0.txt || exit 1;
chmod +rw $dir/lexicon0.txt  # fix a strange permission in the source.
patch <local/dict.patch $dir/lexicon0.txt || exit 1;
grep -v '^#' $dir/lexicon0.txt | awk 'NF>0' | sort > $dir/lexicon1.txt || exit 1;

cp local/MSU_single_letter.txt $dir/
cat $dir/lexicon1.txt $dir/MSU_single_letter.txt  > $dir/lexicon2.txt || exit 1;

# Get the set of lexicon units without noises
# cut -d' ' -f2- $dir/lexicon2.txt | tr ' ' '\n' | sort -u > $dir/units_nosil.txt
cat $dir/lexicon2.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}' | \
  grep -v sil | sort -u  > $dir/units_nosil.txt|| exit 1;

# Add the noises etc. to the lexicon. No silence is added.
(echo '[vocalized-noise] spn'; echo '[noise] nsn'; echo '[laughter] lau'; echo '<unk> spn'; ) | \
 cat - $dir/lexicon2.txt | sort | uniq > $dir/lexicon3.txt || exit 1;

local/swbd1_map_words.pl -f 1 $dir/lexicon3.txt | sort -u > $dir/lexicon4.txt || exit 1;
python local/format_acronyms_dict.py -i $dir/lexicon4.txt -o $dir/lexicon5.txt \
  -L $dir/MSU_single_letter.txt -M $dir/acronyms_raw.map
cat $dir/acronyms_raw.map | sort -u > $dir/acronyms.map
( echo 'i ay' )| cat - $dir/lexicon5.txt | tr '[A-Z]' '[a-z]' | sort -u > $dir/lexicon6.txt

cat $dir/lexicon6.txt | \
  perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; print $_;}}' \
  > $dir/lexicon.txt || exit 1;

# The complete set of lexicon units, indexed by numbers starting from 1
(echo 'spn'; echo 'nsn'; echo 'lau';) | cat - $dir/units_nosil.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Phoneme-based dictionary preparation succeeded"
