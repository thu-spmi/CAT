# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for Spanish.

# Generating lexicon
dict_dir=$1
  g2ps=g2ps/models  # The path containing G2P models from https://github.com/uiuc-sst/g2ps
  phonetisaurus-apply --model $g2ps/spanish_4_3_2.fst --word_list $dict_dir/word_list > $dict_dir/lexicon.txt

# Lexicon correction
cat $dict_dir/lexicon.txt | awk '{$1=""; print $0}' | sed -e 's/g/ɡ/g' > $dict_dir/phone.txt
