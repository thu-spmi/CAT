# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for Indonesian.

# Generating lexicon
dict_dir=$1
  g2ps=local/g2ps/models 
  phonetisaurus-apply --model $g2ps/Indonesian.fst --word_list $dict_dir/word_list > $dict_dir/lexicon.txt

# Lexicon correction
cat $dict_dir/lexicon.txt | awk '{$1=""; print $0}' > $dict_dir/phone.txt