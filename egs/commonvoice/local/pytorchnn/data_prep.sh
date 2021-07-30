#!/usr/bin/env bash

# This script is adopted from `kaldi/egs/swbd/s5c/local/pytorchnn/data_prep.sh`, 
# modified by Wenjie Peng (wenjayep@gmail.com).

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
#text=data/local/lm/librispeech-lm-norm.txt

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
#train=data/train_de/text
#valid=data/dev_de/text
#test=data/test_de/text
#train_text=$dir/train_text.txt


for x in train dev test; do
    mkdir -p data/$x
    cut -f 2- -d " " data/$x/text_pos > $dir/${x}.txt
done

mv $dir/dev.txt $dir/valid.txt

#cat $train_text | cut -d ' ' -f2- | awk -v text_dir=$dir '{if(NR%2000 == 0) { print >text_dir"/valid.txt"; } else {print;}}' >$train

# Sort and preprocess SWBD dataset
#python3 local/pytorchnn/sort_by_start_time.py --infile $train --outfile $dir/cv.train.sorted.txt
#python3 local/pytorchnn/sort_by_start_time.py --infile $valid --outfile $dir/cv.valid.sorted.txt
#python3 local/pytorchnn/sort_by_start_time.py --infile $test --outfile $dir/cv.test.sorted.txt
#for data in train valid test; do
#  cat $dir/cv.${data}.sorted.txt | cut -d ' ' -f 2- > $dir/$data.txt
##  rm $dir/cv.${data}.sorted.txt
#done

# Process Fisher dataset 
mkdir -p $dir/config
#cat > $dir/config/hesitation_mapping.txt <<EOF
#hmm hum
#mmm um
#mm um
#mhm um-hum
#EOF
#gunzip -c $text | cut -d ' ' -f2- | awk -v text_dir=$dir '{if(NR%2000 == 0) { print >text_dir"/valid.txt"; } else {print;}}' >$dir/train.txt
#gunzip -c $fisher | awk 'NR==FNR{a[$1]=$2;next}{for (n=1;n<=NF;n++) if ($n in a) $n=a[$n];print $0}' \
#  > $dir/train.txt

# Merge training data of SWBD and Fisher (ratio 3:1 to match Kaldi RNNLM's preprocessing)
cat $dir/train.txt $dir/train.txt $dir/train.txt > $dir/train_total.txt
cp $dir/train_total.txt $dir/train.txt
rm $dir/train_total.txt 
#rm $dir/fisher.txt

# Symbol for unknown words
echo "<UNK>" >$dir/config/oov.txt
cp data/lang_bpe/words.txt $dir/
# Make sure words.txt contains the symbol for unknown words
if ! grep -w '<UNK>' $dir/words.txt >/dev/null; then
  n=$(cat $dir/words.txt | wc -l)
  echo "<UNK> $n" >> $dir/words.txt
fi

echo "Data preparation done."
