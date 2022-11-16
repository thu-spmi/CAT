#!/bin/bash
# author: Huahuan Zheng
set -e

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

<<"PARSER"
("audio", type=str, help="Audio data directory.")
("trans", type=str, help="Transcription file.")
("-dest", type=str, default="data/src",
    help="Feature extracted path. default: data/src")
PARSER
eval $(python utils/parseopt.py $0 $*)

[ ! -d $audio ] && {
    echo "No such audio folder: '$audio'"
    exit 1
}
[ ! -f $trans ] && {
    echo "No such transcription file: '$trans'"
    exit 1
}
[[ -d $KALDI_ROOT && -d $KALDI_ROOT/egs/wsj/s5 ]] || {
    echo "kaldi tool at '$KALDI_ROOT' not installed. You could specify it with:"
    echo "KALDI_ROOT=... $0 ..."
    exit 1
}

mkdir -p $dest
export dest=$(readlink -f $dest)
cd $KALDI_ROOT/egs/wsj/s5 && . ./path.sh

echo "> Start meta data extracting"
for dataset in train dev test; do
    dir=$dest/$dataset
    mkdir -p $dir
    # prepare wav.scp
    find $audio/$dataset -name *.wav | sort >$dir/wav.path
    sed -e 's/\.wav//' $dir/wav.path |
        awk -F '/' '{print $NF}' >$dir/utt.list
    paste $dir/{utt.list,wav.path} >$dir/wav.scp

    # prep utt2spk, spk2utt
    cut <$dir/utt.list -c 7-11 >$dir/utt2spk.spk
    paste $dir/{utt.list,utt2spk.spk} >$dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $dir/utt2spk >$dir/spk2utt

    # prep text
    utils/filter_scp.pl -f 1 $dir/utt.list $trans |
        sort -u >$dir/text
    rm $dir/{utt.list,utt2spk.spk,wav.path}
    echo "$dataset extracted at $dir"
done

echo "$0 done"
exit 0
