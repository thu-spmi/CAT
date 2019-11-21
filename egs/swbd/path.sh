export KALDI_ROOT=/home/ouzj02/ankeyu/kaldi-5.3
#[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export PATH=/home/ouzj02/xianghy/flac-1.3.2/src/flac:$PATH
export LC_ALL=C

