export CUDA_VISIBLE_DEVICES=0,1,2,3
export CAT_ROOT=../../
export KALDI_ROOT=/opt/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/ctc-crf:$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
export PATH=$CAT_ROOT/src/ctc_crf/path_weight/build:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export PATH=$PATH:/opt/conda/bin
export LC_ALL=C
