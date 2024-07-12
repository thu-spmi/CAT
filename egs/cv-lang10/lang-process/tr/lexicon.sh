# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for Turkish.

# Generating lexicon
dict_dir=$1
  g2ps=g2ps/models 
  phonetisaurus-apply --model $g2ps/turkish.fst --word_list $dict_dir/word_list > $dict_dir/lexicon.txt

# Lexicon correction
cat $dict_dir/lexicon.txt | awk '{$1=""; print $0}' | sed -e 's/d ʒ/d͡ʒ/g; s/dʒ/d͡ʒ/g; s/t ʃ/t͡ʃ/g; s/tʃ/t͡ʃ/g; s/ɡj/ɡ/g; s/g/ɡ/g; s/â/a/g; s/é/e/g; s/û/u/g; s/*//g; s/ ̇//g; s/[.]//g; s/ë/e/g; s/î/i/g' > $dict_dir/phone.txt
