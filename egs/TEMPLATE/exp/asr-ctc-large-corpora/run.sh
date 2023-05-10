#!/bin/bash
#
# This is an example of processing and training
# ... very large corpora. In this example, we
# ... assume the 'train' set is too large to fit into memory.
set -e -u

dir=$(dirname $0)

[ ! -f $dir/.processed_data.done ] && {
    bash local/data.sh

    python local/prep_wds.py >/dev/null || exit 1

    touch $dir/.processed_data.done
}

# train tokenizer
python utils/pipeline/asr.py \
    $dir/tokenizer \
    --sto 1 || exit 1

# finish following steps
# NOTE:
#     with --ld in train:option, the epoch id will always
#     be 1. However, you can estimate the #epochs
#     according to #steps by
#         #epochs = #steps * batch_size / #total_utts
python utils/pipeline/asr.py \
    $dir --sta 2 || exit 1

echo "$0 done"
exit 0
