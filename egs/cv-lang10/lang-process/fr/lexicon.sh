# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for French.

dict_dir=$1
# Generating lexicon
  g2ps=g2ps/models
  phonetisaurus-apply --model $g2psench_8_4_3.fst --word_list $dict_dir/word_list > $dict_dir/lexicon.txt

# Lexicon correction
cat $dict_dir/lexicon.txt | awk '{$1=""; print $0}' | sed -e 's/w ˈa//g; s/g/ʒ/g;
        s/R/ʁ/g; s/í/i/g; s/ì/i/g; s/ò/o/g; s/ó/o/g; s/ü/u/g; s/ú/u/g; s/ù/u/g; s/á/a/g;
        s/ɑ̃/ɑ/g; s/œ̃/œ/g; s/ɛ̃/ɛ/g; s/ÿ/y/g; s/ë/e/g; s/ɔ̃/ɔ/g;' \
    -e 's/[ ]*$//g; s/^[ ]*//g; s/[ ][ ]*/ /g' > $dict_dir/phone.txt
