# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for Polish.

# Generating lexicon
dict_dir=$1
  g2ps=g2ps/models 
  phonetisaurus-apply --model $g2ps/polish_2_2_2.fst --word_list $dict_dir/word_list > $dict_dir/lexicon.txt

# Lexicon correction
cat $dict_dir/lexicon.txt | awk '{$1=""; print $0}' | sed -e 's/ts/t͡s/g; s/dz/d͡z/g; s/ɖʐ/ɖ͡ʐ/g; s/tʂ/ʈ͡ʂ/g; s/dʑ/d͡ʑ/g; s/tɕ/t͡ɕ/g; s/ɔ̃/ɔ/g; s/ɨ̃/ɨ/g; s/ɛ̃/ɛ/g; s/w̃/w/g; s/ɛ̝/ɛ/g' > $dict_dir/phone.txt
