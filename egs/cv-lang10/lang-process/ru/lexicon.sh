# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com) /
# This script prepares phoneme-based lexicon and corrects it for Russian.

dict_dir=$1
# Generating lexicon
  g2ps=g2ps/models
  phonetisaurus-apply --model $g2ps/russian.fst --word_list $dict_dir/word_list > $dict_dir/lexicon.txt

# Lexicon correction
cat $dict_dir/lexicon.txt | awk '{$1=""; print $0}' > $dict_dir/phone.txt
