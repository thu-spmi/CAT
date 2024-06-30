# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# This script prepares phoneme-based lexicon and corrects it for English.

dict_dir=$1
# Generating lexicon
  g2ps=g2ps/models  # The path containing G2P models from https://github.com/uiuc-sst/g2ps
  phonetisaurus-apply --model $g2ps/american-english.fst --word_list $dict_dir/word_list > $dict_dir/lexicon.txt

# Lexicon correction
cat $dict_dir/lexicon.txt | awk '{$1=""; print $0}' | sed -e 's/ˌ//g; s/l̩/l/g; s/n̩/n/g; s/#//g; s/[.]//g; s/g/ɡ/g; s/ei/e i/g; s/aɪ/a ɪ/g; s/ɔi/ɔ i/g; s/oʊ/o ʊ/g; s/aʊ/a ʊ/g; s/ɔɪ/ɔ ɪ/g; s/ɑɪ/ɑ ɪ/g; s/ɝ/ɜ/g; s/ɚ/ə/g; s/tʃ/t͡ʃ/g; s/dʒ/d͡ʒ/g; s/d ʒ/d͡ʒ/g' > $dict_dir/phone.txt
