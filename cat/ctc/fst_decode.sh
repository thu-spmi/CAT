#!/bin/bash
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# Decoding the logits (obtained by cat/ctc/cal_logit.py)
# ... with FST decoding graph (obtained by utils/tool/build_decoding_graph.sh).
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

("--beam", type=float, default=17.0, help="latgen-faster args: --beam")
("--lattice-beam", type=float, default=8.0, help="latgen-faster args: --lattice-beam")
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
symtab="$graph/r_words.txt"
[ -d $log_dir ] && rm -r $log_dir
[ -d $hyps_dir ] && rm -r $hyps_dir
mkdir -p $log_dir
mkdir -p $hyps_dir
f_out=$out_dir/text_$(basename $graph)_ac${acwt}_lm${lmwt}_wip${wip}.hyp
if [[ ! -f $f_out || $force == "True" ]]; then
    run.pl JOB=1:$nj $log_dir/JOB.log latgen-faster \
        --max-mem=200000000 --minimize=false --allow-partial=true \
        --min-active=200 --max-active=7000 \
        --beam=$beam --lattice-beam=$lattice_beam \
        --acoustic-scale=$(echo "$acwt / $lmwt" | bc -l) --word-symbol-table=$symtab \
        $graph/TLG.fst "ark:$rspecifier/decode.JOB.ark" ark:- \| \
        lattice-scale --acoustic-scale=$acwt --lm-scale=$lmwt ark:- ark:- \| \
        lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
        lattice-best-path --word-symbol-table=$symtab ark:- ark,t:- \| \
        $kaldienv/utils/int2sym.pl -f 2- $symtab '>' $hyps_dir/JOB.hyp

    cat $hyps_dir/*.hyp | sort -k 1,1 >$f_out
else
    echo "$out_dir/text.hyp exists, skip decoding." 1>&2
fi

echo $f_out
echo "$0 done." 1>&2
exit 0
