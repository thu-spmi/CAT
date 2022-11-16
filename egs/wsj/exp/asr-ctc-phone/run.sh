#!/bin/bash
# Author: Huahuan Zheng
set -e -u

dir=$(dirname $0)
# Use a hack to re-use the script
touch $dir/den_lm.fst
bash ../asr-ctc-crf-phone/run.sh

exit 0
