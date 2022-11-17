#!/bin/bash
# Copyright Tsinghua University 2022
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# Script for training n-gram LM
set -u
set -e
<<"PARSER"
("dir", type=str, help="Path to the LM directory.")
("--start-stage", type=int, default=1, help="Start stage of the script.")
("--stop-stage", type=int, default=100, help="Stop stage of the script.")
("-o", "--order", type=int, default=5, help="Max order of n-gram. default: 5")
("--output", type=str, default=None,
    help="Path of output N-gram file. default: [dir]/[order]gram.bin")
("--arpa", action="store_true", help="Store n-gram file as .arpa instead of binary.")
("--prune", type=str, default="", nargs='*',
    help="Prune options passed to KenLM lmplz executable. default: ")
("--type", type=str, default="probing", choices=['trie', 'probing'],
    help="Binary file structure. default: probing")
PARSER
eval $(python utils/parseopt.py $0 $*)

export PATH=$PATH:../../src/bin/
[ ! $(command -v lmplz) ] && echo "command not found: lmplz" && exit 1
[ ! $(command -v build_binary) ] && echo "command not found: build_binary" && exit 1
[ ! -d $dir ] && echo "No such directory: $dir" && exit 1

[ $output == "None" ] && {
    if [ $arpa == "True" ]; then
        export output="$dir/${order}gram.arpa"
    else
        export output="$dir/${order}gram.bin"
    fi
}

# train tokenizer & prepare data
[[ $start_stage -le 1 && $stop_stage -ge 1 ]] &&
    python utils/pipeline/lm.py $dir --sto 1

[[ $start_stage -le 2 && $stop_stage -ge 2 ]] &&
    python utils/pipeline/lm.py $dir --sta 2 --sto 2

# n-gram training
[[ $start_stage -le 3 && $stop_stage -ge 3 ]] && {
    echo "==================== Stage 3 N-gram training ===================="
    read -r f_nn_config f_hyper_config f_readme <<<$(python -c "
import sys;
sys.path.append('$(dirname $0)');
from _constants import *;
print(F_NN_CONFIG, F_HYPER_CONFIG, F_TRAINING_INFO)")
    f_nn_config="$dir/$f_nn_config"
    f_hyper_config="$dir/$f_hyper_config"
    [ ! -f $f_hyper_config ] && echo "No hyper-setting file: $f_hyper_config" && exit 1
    [ ! -f $f_nn_config ] && echo "No model config file: $f_nn_config" && exit 1

    if [ "$prune" ]; then
        prune="--prune $prune"
    fi

    # we need to manually rm the bos/eos/unk since lmplz tool would add them
    # ... and kenlm not support <unk> in corpus,
    # ... so in the `utils/data/corpus2index.py` script we convert 0(<bos>, <eos>) and 1 (<unk>) to white space
    # ... if your tokenizer set different bos/eos/unk id, you should make that mapping too.
    # this step also filter out the utterance id (since utt id is 1 (<unk>), which will be then map to empty).
    train_cmd="python utils/data/corpus2index.py $dir/pkl/train.pkl --map 0: 1: | lmplz -o $order $prune -S 20%"
    [ $arpa == "True" ] && train_cmd="$train_cmd >$output"

    # NOTE: if lmplz raises error telling the counts of n-grams are not enough,
    # you should probably duplicate your text corpus or add the option --discount_fallback
    # Error msg sample:
    # "ERROR: 3-gram discount out of range for adjusted count 3: -5.2525253."
    train_cmd="$train_cmd || $train_cmd --discount_fallback"
    [ $arpa == "False" ] && train_cmd="($train_cmd) | build_binary $type /dev/stdin $output"

    if [ -f $output ]; then
        echo "KenLM $output found, skip lmplz."
    else
        eval $train_cmd
    fi

    if [ -f $f_nn_config ]; then
        export vocab_size="$(cat $f_hyper_config |
            python -c "import sys,json;print(json.load(sys.stdin)['tokenizer']['|V|'])")"
        cat $f_nn_config | python -c "
import sys, json
configure = json.load(sys.stdin)
configure['decoder']['kwargs']['f_binlm'] = '$output'
configure['decoder']['kwargs']['gram_order'] = $order
configure['decoder']['kwargs']['num_classes'] = $vocab_size
json.dump(configure, sys.stdout, indent=4)" >"$f_nn_config.tmp"
        mv "$f_nn_config.tmp" $f_nn_config
    fi

    echo "LM saved at $output."

    f_readme="$dir/$f_readme"
    [ ! -f $f_readme ] && (
        echo -e "\ntrain command:\n" >>$f_readme
        echo -e "\`\`\`bash\n$0 $@\n\`\`\`" >>$f_readme
        echo -e "\nproperty:\n" >>$f_readme
        echo "- prune: $prune" >>$f_readme
        echo "- type:  $type" >>$f_readme
        echo "- size:  $(ls -lh $output | awk '{print $5}')B" >>$f_readme
        echo -e "\nperplexity:\n" >>$f_readme
        echo -e "\`\`\`\n\n\`\`\`" >>$f_readme
    )
}

[[ $start_stage -le 4 && $stop_stage -ge 4 ]] &&
    python utils/pipeline/lm.py $dir --sta 4 --sto 4

echo "$0 done."
exit 0
