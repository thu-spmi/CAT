#!/bin/bash
# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)
#
# Decode the logits (obtained by cat/ctc/cal_logit.py) with
# ... FST decoding graph (obtained by utils/tool/build_decoding_graph.sh).
set -e

<<"PARSER"
("rspecifier", type=str,
    help="Directory of logits in ark format. Commonly like decode.1.ark, ...")
("graph", type=str,
    help="Directory of decoding graph, where TLG.fst and r_words.txt are expected.")
("out_dir", type=str,
    help="Output directory, where lattice, hypotheses and logs would be saved.")
("-f", "--force", action="store_true", default=False,
    help="Force to do the decoding whatever the result exists or not.")
("--nj", type=int, default=-1, help="Number of jobs. This should match the nj of cal_logit.py")
("--acwt", type=float, default=1.0, help="AC score weight. default: 1.0")
("--lmwt", type=float, default=1.0, help="LM score weight. default: 1.0")
("--wip", type=float, default=0.0, help="Word insertion penalty factor. default: 0.0")

("--beam", type=int, default=17, help="latgen-faster args: --beam")
("--lattice-beam", type=int, default=8, help="latgen-faster args: --lattice-beam")
("-n", "--nbest", type=int, default=16, help="Number of output nbest lists per utterance.")
PARSER
eval $(python utils/parseopt.py $0 $*)

[ $nj == "-1" ] && {
    nj=$(ls $rspecifier/decode.*.ark | wc -l)
}

export PATH=$PATH:../../src/bin/
[ ! $(command -v latgen-faster) ] && {
    echo "command not found: latgen-faster" 1>&2
    echo "install with:" 1>&2
    echo "  cd $(readlink -f $(pwd)/../..)" 1>&2
    echo "  ./install.sh -f fst-decoder" 1>&2
    exit 1
}

[ -z $KALDI_ROOT ] && {
    echo "\$KALDI_ROOT is not set. re-run with" 1>&2
    echo "KALDI_ROOT=xxx $0 $*" 1>&2
    exit 1
}
export KALDI_ROOT=$KALDI_ROOT
kaldienv="$KALDI_ROOT/egs/wsj/s5"
cd $kaldienv && . ./path.sh && cd - >/dev/null

log_dir="$out_dir/log"
hyps_dir="$out_dir/hyp"
lat_dir="$out_dir/lat"
symtab="$graph/r_words.txt"
f_out=$out_dir/ac${acwt}_lm${lmwt}_wip${wip}.hyp
if [[ ! -f $f_out || $force == "True" ]]; then
    mkdir -p $log_dir
    mkdir -p $hyps_dir
    mkdir -p $lat_dir
    # obtain lattice
    run.pl JOB=1:$nj $log_dir/lat.JOB.log latgen-faster \
        --max-mem=200000000 --minimize=false --allow-partial=true \
        --min-active=200 --max-active=7000 \
        --beam=$beam --lattice-beam=$lattice_beam \
        --acoustic-scale=$(echo "$acwt / $lmwt" | bc -l) --word-symbol-table=$symtab \
        $graph/TLG.fst "ark:$rspecifier/decode.JOB.ark" ark:- \| \
        lattice-scale --acoustic-scale=$acwt --lm-scale=$lmwt ark:- ark:- \| \
        lattice-add-penalty --word-ins-penalty=$wip ark:- ark:$lat_dir/lat.JOB.ark

    # obtain transcript
    run.pl JOB=1:$nj $log_dir/text.JOB.log \
        lattice-best-path --word-symbol-table=$symtab ark:$lat_dir/lat.JOB.ark ark,t:- \| \
        $kaldienv/utils/int2sym.pl -f 2- $symtab '>' $hyps_dir/JOB.hyp
    cat $hyps_dir/*.hyp | sort -k 1,1 >$f_out

    # lattice to nbest list
    run.pl JOB=1:$nj $log_dir/lat2nbest.JOB.log \
        lattice-to-nbest --n=$nbest ark:$lat_dir/lat.JOB.ark ark:- \| \
        nbest-to-linear ark:- ark:/dev/null ark,t:- \
        ark,t:$lat_dir/lm_cost.JOB.txt ark,t:$lat_dir/ac_cost.JOB.txt \| \
        int2sym.pl -f 2- $symtab \| \
        sed "'s/<UNK>//g'" ">" $lat_dir/trans.JOB.txt

    for file in trans lm_cost ac_cost; do
        for i in $(seq 1 1 $nj); do
            cat $lat_dir/$file.$i.txt
        done | sort -k 1 >$lat_dir/$file.txt
        rm $lat_dir/$file.*.txt
    done
    # python utils/data/text2nbest.py $lat_dir/{trans.txt,ac_cost.txt} $f_out.nbest

else
    echo "$f_out exists, skip decoding." 1>&2
fi

echo $f_out
echo "$0 done." 1>&2
# rm -r $log_dir $hyps_dir $lat_dir
exit 0
