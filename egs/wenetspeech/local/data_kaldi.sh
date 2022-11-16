#!/bin/bash
# author: Huahuan Zheng
# this script prepare the data by kaldi tool
set -e -u

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

<<"PARSER"
("-src", type=str, default="/data/WenetSpeech/",
    help="Source data folder containing the audios and transcripts. ")
PARSER
eval $(python utils/parseopt.py $0 $*)

[ -z $KALDI_ROOT ] && {
    echo "\$KALDI_ROOT is not set. re-run with"
    echo "KALDI_ROOT=xxx $0 $*"
    exit 1
}
export KALDI_ROOT=$KALDI_ROOT

export LC_ALL="zh_CN.UTF-8"
corpus_dir="data/corpus"
[ ! -f $corpus_dir/.done ] && {
    [ ! -d $corpus_dir ] && mkdir -p $corpus_dir
    python3 local/extract_meta.py --pipe-format \
        $src/WenetSpeech.json $corpus_dir || exit 1
    touch $corpus_dir/.done
}

script=$(readlink -f local/wenetspeech_data_prep.sh)
srcdir=$(readlink -f $src)
dstdir=$(readlink -f data/src)
corpus_dir=$(readlink -f $corpus_dir)
cd $KALDI_ROOT/egs/wsj/s5
bash $script $srcdir $dstdir $corpus_dir
cd - >/dev/null

bash utils/data/data_prep_kaldi.sh \
    data/src/{train_m,dev,test_net,test_meeting} \
    --feat-dir=data/fbank \
    --nj=48 \
    --not-apply-cmvn

# refresh the data/metainfo.json file
python utils/data/resolvedata.py

echo "$0 done."
exit 0
