#!/bin/bash
# Author: Huahuan Zheng (maxwellzh@outlook.com)
set -e -u
<<"PARSER"
("dir", type=str, help="Experiment directory.")
("graph", type=str, 
    help="Decoding graph directory, where TLG.fst and r_words.txt are expected.")
("--data", type=str, nargs='+', required=True,
    help="Dataset(s) to be evaluated. e.g. dev test")
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
("-n", "--nbest", type=int, default=16, help="Number of output nbest lists per utterance.")
PARSER
eval $(python utils/parseopt.py $0 $*)

function get_test_tran() {
    echo $(
        python -c \
            "import json;print(json.load(open('data/metainfo.json'))['$1']['trans'])"
    )
}

opt_force=""
[ $force == "True" ] &&
    opt_force="-f"

opt_er=""
[ $mode == "cer" ] &&
    opt_er="--cer"

cache="/tmp/$(
    head /dev/urandom | tr -dc A-Za-z0-9 | head -c10
).log"
for set in $data; do
    fout=$(bash cat/ctc/fst_decode.sh $opt_force \
        --acwt $acwt --lmwt $lmwt \
        --wip $wip --nbest $nbest \
        $dir/decode/$set/ark $graph $dir/decode/$set)

    # echo -en "${set}_$(basename $fout)\t"
    # python utils/wer.py --cer \
    #     $(get_test_tran $set) $fout
    echo -en "${set}_$(basename $fout)\t"
    python utils/wer.py  \
        $(get_test_tran $set) $fout

done 2>$cache || {
    cat $cache
    exit 1
}
rm $cache

exit 0
