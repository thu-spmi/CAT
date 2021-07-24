#!/bin/bash

# This script comes from Kaldi. It retrains the language models using the expanded
# lexicon/vocabulary.
# It demonstrates how to build LMs using the kaldi_lm, SRILM, IRSTLM toolkits. Although
# we use the LMs built by kaldi_lm in the subsequent steps, LMs built by the other two
# toolkits can be used as well.

. ./path.sh
. ./cmd.sh

text=$1 # data/train/text  
srcdir=$2  # data/local/dict_phn
dir=$3 # data/local/lm_char
mkdir -p $dir
unk_id=2

set -euo pipefail

( # First make sure the kaldi_lm toolkit is installed.
    cd $KALDI_ROOT/tools/ || exit 1;
    if [ -d kaldi_lm ]; then
        echo Not installing the kaldi_lm toolkit since it is already there.
    else
        echo Downloading and installing the kaldi_lm tools
        if [ ! -f kaldi_lm.tar.gz ]; then
        wget http://www.danielpovey.com/files/kaldi/kaldi_lm.tar.gz || exit 1;
        fi
        tar -xvzf kaldi_lm.tar.gz || exit 1;
        cd kaldi_lm
        make || exit 1;
        echo Done making the kaldi_lm tools
    fi
) || exit 1;

# Get a wordlist-- keep everything but silence, which should not appear in
# the LM.
awk '{print $1}' $srcdir/lexicon.txt > $dir/wordlist.txt

# Get training data without OOVs
cat $text | awk -v w=$dir/wordlist.txt \
  'BEGIN{while((getline<w)>0) v[$1]=1;}
  {for (i=1;i<=NF;i++) if ($i in v) printf $i" ";else printf "<SPN> " ;print ""}'|sed 's/ $//g' \
  | gzip -c > $dir/train_nounk.gz


# Get unigram counts (without bos/eos, but this doens't matter here, it's
# only to get the word-map, which treats them specially & doesn't need their
# counts).
# Add a 1-count for each word in word-list by including that in the data,
# so all words appear.
gunzip -c $dir/train_nounk.gz | cat - $dir/wordlist.txt | \
  awk '{ for(x=1;x<=NF;x++) count[$x]++; } END{for(w in count){print count[w], w;}}' | \
 sort -nr > $dir/unigram.counts

# Get "mapped" words-- a character encoding of the words that makes the common words very short.
# <unk>=2, <s>=0, </s>=1
cat $dir/unigram.counts  | awk '{print $2}' | get_word_map.pl "<s>" "</s>"  "<SPN>" > $dir/word_map

# we don't have oovs
gunzip -c $dir/train_nounk.gz | awk -v wmap=$dir/word_map 'BEGIN{while((getline<wmap)>0)map[$1]=$2;}
  { for(n=1;n<=NF;n++) { printf map[$n]; if(n<NF){ printf " "; } else { print ""; }}}' | gzip -c >$dir/train.gz

# To save disk space, remove the un-mapped training data.  We could
# easily generate it again if needed.
rm $dir/train_nounk.gz 

train_lm.sh --arpa --lmtype 3gram-mincount $dir
echo "Done with training lms"

prune_lm.sh --arpa 6.0 $dir/3gram-mincount/

train_lm.sh --arpa --lmtype 4gram-mincount $dir

prune_lm.sh --arpa 7.0 $dir/4gram-mincount

### Below here, this script is showing various commands that 
## were run during LM tuning.

train_lm.sh --arpa --lmtype 3gram-mincount $dir

prune_lm.sh --arpa 3.0 $dir/3gram-mincount/

prune_lm.sh --arpa 6.0 $dir/3gram-mincount/

train_lm.sh --arpa --lmtype 4gram-mincount $dir

prune_lm.sh --arpa 3.0 $dir/4gram-mincount

prune_lm.sh --arpa 4.0 $dir/4gram-mincount

prune_lm.sh --arpa 5.0 $dir/4gram-mincount

prune_lm.sh --arpa 7.0 $dir/4gram-mincount

train_lm.sh --arpa --lmtype 3gram $dir
train_lm.sh --arpa --lmtype 4gram $dir

exit 0;
