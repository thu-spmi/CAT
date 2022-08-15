#!/bin/bash

# To be run from one directory above this script.

. ./path.sh

text=$1
lexicon=$2
dir=$3
for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

#text=data/train/text
#lexicon=data/dict/lexicon.txt
#dir=data/lm
mkdir -p $dir

cleantext=$dir/text.no_oov

cat $text | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleantext || exit 1;

cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | sort | uniq -c | \
   sort -nr > $dir/word.counts || exit 1;

# Get counts from acoustic training transcripts, and add  one-count
# for each word in the lexicon (but not silence, we don't want it
# in the LM-- we'll add it optionally later).
cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | \
  cat - <(grep -w -v '!SIL' $lexicon | awk '{print $1}') | \
   sort | uniq -c | sort -nr > $dir/unigram.counts || exit 1;

# note: we probably won't really make use of <UNK> as there aren't any OOVs
cat $dir/unigram.counts  | awk '{print $2}' | ${H}/local/get_word_map.pl "<s>" "</s>" "<UNK>" > $dir/word_map \
   || exit 1;

# note: ignore 1st field of train.txt, it's the utterance-id.
cat $cleantext | awk -v wmap=$dir/word_map 'BEGIN{while((getline<wmap)>0)map[$1]=$2;}
  { for(n=2;n<=NF;n++) { printf map[$n]; if(n<NF){ printf " "; } else { print ""; }}}' | gzip -c >$dir/train.gz \
   || exit 1;

# LM is small enough that we don't need to prune it (only about 0.7M N-grams).

# From here is some commands to do a baseline with SRILM (assuming
# you have it installed).
heldout_sent=3 
sdir=$dir/srilm
mkdir -p $sdir
cat $cleantext | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' | \
  head -$heldout_sent > $sdir/heldout
cat $cleantext | awk '{for(n=2;n<=NF;n++){ printf $n; if(n<NF) printf " "; else print ""; }}' | \
  tail -n +$heldout_sent > $sdir/train

cat $dir/word_map | awk '{print $1}' | cat - <(echo "<s>"; echo "</s>" ) > $sdir/wordlist

ngram-count -text $sdir/train -order 1 -limit-vocab -vocab $sdir/wordlist -unk \
  -map-unk "<UNK>" -interpolate -lm $sdir/srilm.o1g.kn.gz
# -kndiscount
ngram -lm $sdir/srilm.o1g.kn.gz -ppl $sdir/heldout 
