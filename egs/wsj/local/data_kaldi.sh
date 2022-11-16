#!/bin/bash
# author: Huahuan Zheng
# this script prepare the aishell data by kaldi tool
# NOTE: in this script we defaultly apply CMVN.
set -e

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

<<"PARSER"
("wsj0", type=str, nargs='?', default="/data/wsj/csr_1",
    help="Source data folder WSJ0.")
("wsj1", type=str, nargs='?', default="/data/wsj/csr_2_comp",
    help="Source data folder WSJ1.")
("-use-3way-sp", action="store_true",
    help="Use 3-way speed perturbation.")
PARSER
eval $(python utils/parseopt.py $0 $*)

[[ ! -d $wsj0 || ! -d $wsj1 ]] && {
    echo "At least one of the given folder does not exist:"
    echo "   1. $wsj0"
    echo "   2. $wsj1"
    exit 1
}
[ -z $KALDI_ROOT ] && {
    echo "\$KALDI_ROOT is not set. re-run with"
    echo "KALDI_ROOT=xxx $0 $*"
    exit 1
}
export KALDI_ROOT=$KALDI_ROOT

if [ $use_3way_sp == "True" ]; then
    opt_3way_sp="--apply-3way-speed-perturbation"
else
    opt_3way_sp=""
fi

[ ! -f data/src/.done ] && {
    # use kaldi recipe to prepare the meta data.
    mkdir -p .cache/wsj
    cp -r $KALDI_ROOT/egs/wsj/s5/* .cache/wsj/ >/dev/null
    
    cd .cache/wsj
    # skip setting the kaldi root
    sed -i "s/^export KALDI_ROOT=.*$//g" path.sh
    local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?
    cd - >/dev/null

    srcdir=.cache/wsj/data/local/data
    dstdir=data/src
    mkdir -p $dstdir
    for x in train_si284 test_eval92 test_dev93; do
        mkdir -p $dstdir/$x
        cp $srcdir/${x}_wav.scp $dstdir/$x/wav.scp
        cp $srcdir/$x.txt $dstdir/$x/text
        cp $srcdir/$x.spk2utt $dstdir/$x/spk2utt
        cp $srcdir/$x.utt2spk $dstdir/$x/utt2spk
    done
    rm -rf .cache/wsj
    touch data/src/.done
    echo "Meta data extracted."
}


# prepare extra corpus
[ ! -f data/extra.corpus ] &&
    zcat $wsj1/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z |
    grep -v "<" |
        tr "[:lower:]" "[:upper:]" |
        >data/extra.corpus

# compute fbank feat
# by default, we use 80-dim raw fbank and do not apply
# ... the CMVN, which matches the local/data.sh via torchaudio
bash utils/data/data_prep_kaldi.sh \
    data/src/{train_si284,test_dev93,test_eval92} \
    --feat-dir=data/fbank \
    --nj=$(nproc) \
    $opt_3way_sp

# prepare cmu dict in case of usage
[ ! -f data/cmudict.txt ] &&
    wget http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict.0.7a \
        -O - | grep -v ";;;" >data/cmudict.txt

# refresh the data/metainfo.json file
python utils/data/resolvedata.py

echo "$0 done."
echo "go and check data/metainfo.json for dataset info."
echo "CMU dict: data/cmudict.txt"
exit 0
