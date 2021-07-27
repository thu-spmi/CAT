#!/usr/bin/env bash

# This script is adopted from `kaldi/egs/swbd/s5c/local/pytorchnn/data_prep.sh`, modified by
# Wenjie Peng (wenjayep@gmail.com). Note: we don't need to prepare data since we can use the 
# RWTH pretrained Transformer LM.

# This script prepares the data directory for PyTorch based neural LM training.
# It prepares the following files in a output directory:
# 1. Vocabulary: $dir/words.txt copied from data/lang/words.txt.
# 2. Training and test data: $dir/{train/valid/test}.txt with each sentence per line.
#    Note: train.txt contains both training data of SWBD and Fisher. And the train
     # and dev datasets of SWBD are not the same as Kaldi RNNLM as we use
#    data/train_nodev/text as training data and data/train_dev/text as valid data.
#    While Kaldi RNNLM split data/train_nodev/text as train/dev for SWBD.
#    The test.txt can be any test set users are interested in, for example, eval2000.
#    We sorted utterances of each conversation of SWBD as we found it gives
#    better perplexities and WERs.


# Begin configuration section.
stage=0
text=data/local/lm/librispeech-lm-norm.txt

#valid=data/train_cv05/text
#test=data/test_clean/text
#text=data/local/lm/librispeech-lm-norm.txt

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

set -e

if [ $# != 1 ]; then
   echo "Usage: $0 <dest-dir>"
   echo "For details of what the script does, see top of script file"
   exit 1;
fi

dir=$1 # data/pytorchnn/
mkdir -p $dir
train=$dir/train.txt
valid=$dir/valid.txt
test=$dir/test.txt
text=data/local/lm/librispeech-lm-norm.txt
echo -n > $train

for x in tr95 cv05; do
    cut -f 2- -d ' ' data/train_$x/text >> $train
done

cat $text | grep -v '^$' >> $train

echo -n > $test
echo -n > $valid

for x in dev_clean dev_other; do
  cut -f 2- -d ' ' data/$x/text >> $valid
done

for x in test_clean test_other; do
  cut -f 2- -d ' ' data/$x/text >> $test
done
#for f in $train $valid $test $fisher; do
#    [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1
#done

# Sort and preprocess SWBD dataset
#python3 local/pytorchnn/sort_by_start_time.py --infile $train --outfile $dir/swbd.train.sorted.txt
#python3 local/pytorchnn/sort_by_start_time.py --infile $valid --outfile $dir/swbd.valid.sorted.txt
#python3 local/pytorchnn/sort_by_start_time.py --infile $test --outfile $dir/swbd.test.sorted.txt
#for data in train valid test; do
#  cat $dir/swbd.${data}.sorted.txt | cut -d ' ' -f2- > $dir/$data.txt
#  rm $dir/swbd.${data}.sorted.txt
#done

# Process Fisher dataset 
mkdir -p $dir/config
#gunzip -c $text | cut -d ' ' -f2- | awk -v text_dir=$dir '{if(NR%2000 == 0) { print >text_dir"/valid.txt"; } else {print;}}' >$dir/train.txt
#gunzip -c $fisher | awk 'NR==FNR{a[$1]=$2;next}{for (n=1;n<=NF;n++) if ($n in a) $n=a[$n];print $0}' \
#  > $dir/train.txt

# Merge training data of SWBD and Fisher (ratio 3:1 to match Kaldi RNNLM's preprocessing)
#cat $dir/train.txt $dir/fisher.txt $dir/train.txt $dir/train.txt > $dir/train_total.txt
#cp $dir/train.txt $dir/train_total.txt
#rm $dir/train.txt
#mv $dir/train_total.txt $dir/train.txt
#rm $dir/fisher.txt

# Symbol for unknown words
echo "<UNK>" >$dir/config/oov.txt
cp data/lang_phn/words.txt $dir/
# Make sure words.txt contains the symbol for unknown words
if ! grep -w '<UNK>' $dir/words.txt >/dev/null; then
  n=$(cat $dir/words.txt | wc -l)
  echo "<UNK> $n" >> $dir/words.txt
fi

echo "Data preparation done."
