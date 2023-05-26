#!/bin/bash
set -e -u

dir=$(dirname $0)

bash local/lm_data.sh

python utils/pipeline/lm.py $dir --sto 3 || exit 1

echo "$0 done" && exit 0
