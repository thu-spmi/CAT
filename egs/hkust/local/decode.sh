# Begin configuration section.
stage=1
nj=20 # number of decoding jobs.  If --transform-dir set, must match that number!
cmd=run.pl
beam=17.0
max_active=7000
min_active=200
max_mem=200000000 # approx. limit to memory consumption during minimization in bytes
lattice_beam=8.0 # Beam we use in lattice generation.
scoring_opts=
minimize=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.


acwt=$1  # Just a default value, used for adaptation and beam-pruning.
dataset=$2
nj=$3

if [[ $# = 4 ]];then
  lmtype=$4
  dir=exp/decode_${dataset}/lattice_$lmtype
  graphdir=data/lang_phn_test_$lmtype
else
  dir=exp/decode_${dataset}/lattice
  graphdir=data/lang_phn_test
fi



data=data/dev
ark_dir=exp/decode_${dataset}/ark


mkdir -p $dir
mkdir -p $ark_dir

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $ark_dir/log/decode.JOB.log \
    latgen-faster --max-mem=$max_mem --min-active=$min_active --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
    --minimize=$minimize --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $graphdir/TLG.fst "ark:$ark_dir/decode.JOB.ark" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1
fi

if [ $stage -le 2 ]; then
    ./local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir 
    echo "score confidence and timing with sclite"
fi
echo "Decoding done."

