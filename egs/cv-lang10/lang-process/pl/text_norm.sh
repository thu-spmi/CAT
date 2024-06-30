# Copyright 2023 Tsinghua SPMI Lab, Author: Ma, Te (mate153125@gmail.com)
# Acknowlegement: This script refer to the code of Huahuan Zheng (maxwellzh@outlook.com)
# This script completes text normalization for Polish dataset from CommonVoice

data_dir=$1
    for set in dev test excluded_train; do
      paste $data_dir/$set/text > $data_dir/$set/text.bak
      cut <$data_dir/$set/text.bak -f 2- | \
         sed -e 's/,/ /g; s/"/ /g; s/“/ /g; s/[;]/ /g; s/[—]/ /g; s/[.]/ /g; s/:/ /g; s/!/ /g; s/”/ /g; s/?/ /g; s/«/ /g; s/»/ /g' | \
         sed -e 's/[ ][ ]*/ /g; s/^[ ]*//g; s/[ ]*$//g' | \
         python -c "import sys; print(sys.stdin.read().lower())" > $data_dir/$set/text.trans.tmp
      cut <$data_dir/$set/text.bak -f 1 > $data_dir/$set/text.id.tmp
      paste $data_dir/$set/text.{id,trans}.tmp > $data_dir/$set/text
      cat $data_dir/$set/text | sed -e 's/^[	]*//g' | grep -v "^$" > $data_dir/$set/text_new
      mv $data_dir/$set/text_new $data_dir/$set/text
      rm -rf $data_dir/$set/text.{id,trans}.tmp
    done