#!/usr/bin/env bash
#
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#           2020  Alex Hung (hung_alex@icloud.com)
# Apache 2.0

cmd=run.pl
# Begin configuration section.
stage=0
nj=4
beam=17.0
acwt=1.0
post_decode_acwt=10.0 # see https://kaldi-asr.org/doc/chain.html#chain_decoding
max_active=7000
min_active=200
max_mem=200000000 # approx. limit to memory consumption during minimization in bytes
lattice_beam=8.0 # Beam we use in lattice generation.
scoring_opts=
minimize=false
model=best_model
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: ctc-crf/decode.sh [options] <graph-dir> <data-dir> <input-scp> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: ctc-crf/decode.sh data/lang_phn_test data/test data/test_data/test.scp exp/TDNN/decode"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --model <string:best_model>                      # which model to use."
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --acwt <float:1.0>                               # acoustic scale used for lattice generation "
   echo "  --scoring-opts <string>                          # options to local/score.sh"
   exit 1;
fi

graph=$1
data=$2
input_scp=$3
dir=$4
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.

logits=$dir/logits
TLG_dir=$graph

[ ! -d $dir ] && mkdir -p $dir
echo "$nj" > $dir/num_jobs

# check files
num_error=0
for f in $TLG_dir/TLG.fst $graph/words.txt $input_scp $srcdir/$model $srcdir/config.json; do
  if [ ! -f "$f" ]; then
    echo "Missing file: $f"
    num_error=$((num_error + 1))
  fi
done
[ $num_error -gt 0 ] && exit 1;

if [ $stage -le 0 ]; then
  mkdir -p $logits
  python3 ctc-crf/calculate_logits.py \
    --nj=$nj \
    --input_scp=$input_scp \
    --config=$srcdir/config.json \
    --model=$srcdir/$model \
    --output_dir=$logits
fi

if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="ark:|gzip -c >$dir/lat.JOB.gz"
else
  lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
fi

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    latgen-faster --max-mem=$max_mem --min-active=$min_active --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
    --minimize=$minimize --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graph/words.txt \
    $TLG_dir/TLG.fst "ark:$logits/decode.JOB.ark" "$lat_wspecifier" || exit 1
fi

if [ $stage -le 2 ]; then
    local/score.sh $scoring_opts --cmd "$cmd" $data $graph $dir
    echo "score confidence and timing with sclite"
fi
echo "Decoding done."
