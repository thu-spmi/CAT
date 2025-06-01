#!/bin/bash
# Copyright 2025 Tsinghua SPMI Lab
# Author: Sardar (sar_dar@foxmail.com)

set -e -u
<<"PARSER"
("dir", type=str, help="Experiment directory.")
("graph", type=str, 
    help="Decoding graph directory, where TLG.fst and r_words.txt are expected.")
("--acwt", type=float, default=1.0, 
    help="AC score weight.")
("--lmwt", type=float, default=1.0, 
    help="LM score weight.")
("--wip", type=float, default=0.0, 
    help="Word insertion penalty.")
("--mode", type=str, choices=['cer', 'wer'], default='wer',
    help="Evaluate with wer or cer..")
("-f", "--force", action="store_true", default=False,
    help="Force to do the decoding whatever the result exists or not.")
("--beam", type=int, default=17, help="latgen-faster args: --beam")
("--lattice-beam", type=int, default=8, help="latgen-faster args: --lattice-beam")
("--mls", action="store_true", default=False,
    help="Wheter use nbest list for MLS decoding.")
("-n", "--nbest", type=int, default=16, help="Number of output nbest lists per utterance.")
PARSER
eval $(python utils/parseopt.py $0 $*)

function get_test_tran() {
    echo $(
        python -c \
            "import json;print(json.load(open('data/metainfo.json'))['$1']['trans'])"
    )
}
function get_test_set() {
    echo $(
        python -c \
            "import json;print(json.load(open('$1/hyper-p.json'))['data']['test'][0])"
    )
}
opt_force=""
[ $force == "True" ] &&
    opt_force="-f"

set=$(get_test_set $dir)
fout=$(bash cat/ctc/fst_decode.sh $opt_force \
    --acwt $acwt --lmwt $lmwt \
    --wip $wip --nbest $nbest \
    --beam $beam --lattice-beam ${lattice_beam} \
    $dir/decode/$set/ark $graph $dir/decode/$set)

echo -en "${set}_$(basename $fout)\t"

opt_er=""
[ $mode == "cer" ] &&
    opt_er="--cer"

python utils/wer.py  $opt_er \
    $(get_test_tran $set) $fout

if [[ $mls == "True" ]]; then
    python utils/data/text2nbest.py $dir/decode/$set/lat/{trans.txt,lm_cost.txt} ${dir}/decode/${set}/ac1.0_lm${lmwt}.n64.nbest
fi

rm -r -f $dir/decode/$set/{log,lat,hyp}
exit 0
