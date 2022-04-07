# CAT toolkit
export CAT_ROOT=../../
export PATH=$CAT_ROOT/src/ctc_crf/path_weight/build:$PATH
export PATH=$PWD/ctc-crf:$PATH
# Kaldi
export KALDI_ROOT=${KALDI_ROOT:-/opt/kaldi}
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
# Data
export DATA_ROOT=/mnt/nas_workspace2/spmiData/THCHS30
