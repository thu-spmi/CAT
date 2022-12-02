# prepare spec feats by kaldi tool
# author: Huahuan Zheng (maxwellzh@outlook.com)
set -e
<<"PARSER"
("data_dir", type=str, nargs='+',
    help="Directory of data (containing wav.scp, utt2spk, ...)")
("--feat-dir", type=str, required=True,
    help="Ouput directory of features. This only includes the raw features. "
    "The .scp files would be stored as $data_dir/feats.scp | $data_dir/feats_cmvn.scp")
("--fbank-conf", type=str,
    help="Specify the configuration file for extracting FBank features. "
    "If not set, extract raw 80-dim.")
("--apply-3way-speed-perturbation", action="store_true",
    help="Enable 3way speed perturbation (0.9, 1.0, 1.1).")
("--not-apply-cmvn", action="store_true", help="Disable applying CMVN.")
("--nj", type=int, default=16, help="Number of jobs. default: 16")
("--kaldi-root", type=str, help="Path to kaldi folder. Not required if $KALDI_ROOT is set.")
PARSER
eval $(python utils/parseopt.py $0 $*)

# generate fbank features
if [ $fbank_conf == "None" ]; then
    echo "--num-mel-bins=80" >>fb.conf.tmp
else
    cp $fbank_conf fb.conf.tmp
fi
export fbank_conf=$(readlink -f fb.conf.tmp)

if [ $kaldi_root != "None" ]; then
    export KALDI_ROOT=$kaldi_root
else
    [ -z $KALDI_ROOT ] && (
        echo "\$KALDI_ROOT is not specified."
        exit 1
    )
fi
! [[ -d $KALDI_ROOT && -d $KALDI_ROOT/egs/wsj/s5 ]] && (
    echo "kaldi tool at '$KALDI_ROOT' not installed."
    exit 1
)

export feat_dir=$(readlink -f $feat_dir)
export data_dir=$(readlink -f $data_dir)
cd $KALDI_ROOT/egs/wsj/s5 && . ./path.sh

[ $apply_3way_speed_perturbation == "True" ] && {
    all_sp_dir=""
    for dir in $data_dir; do
        sp_dir="$(echo $dir | sed -e 's/[/]$//g')-3sp"
        [ ! -f $sp_dir/feats.scp ] &&
            utils/data/perturb_data_dir_speed_3way.sh $dir $sp_dir
        all_sp_dir="$all_sp_dir $sp_dir"
    done
    data_dir="$data_dir $all_sp_dir"
}

mkdir -p $feat_dir
for dir in $data_dir; do
    if [ ! -f $dir/.feats.done ]; then
        utils/fix_data_dir.sh $dir
        steps/make_fbank.sh --cmd run.pl --nj $nj \
            --fbank-config $fbank_conf \
            $dir \
            $feat_dir/log \
            $feat_dir/fbank ||
            exit 1
        touch $dir/.feats.done
    else
        echo "Found previous features, if you want to re-compute anyway,"
        echo "... remove the '$dir/.feats.done' then re-run this script."
    fi

    if [ $not_apply_cmvn == "False" ]; then
        if [ -f $dir/feats_cmvn.scp ]; then
            [ -f $dir/feats.scp ] && {
                if [ -f $dir/feats_orin.scp ]; then
                    rm -f $dir/feats.scp
                else
                    mv $dir/feats.scp $dir/feats_orin.scp
                fi
            }
            ln -snf feats_cmvn.scp $dir/feats.scp
        else
            [ ! -f $dir/feats_orin.scp ] &&
                mv $dir/feats.scp $dir/feats_orin.scp
            ln -snf feats_orin.scp $dir/feats.scp

            steps/compute_cmvn_stats.sh \
                $dir \
                $feat_dir/log-cmvn \
                $feat_dir/cmvn ||
                exit 1

            copy-feats --compress=true \
                "ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$dir/utt2spk \
                scp:$dir/cmvn.scp scp:$dir/feats_orin.scp ark:- |" \
                "ark,scp:$dir/applied_cmvn.ark,$dir/feats_cmvn.scp"

            ln -snf feats_cmvn.scp $dir/feats.scp
        fi
    else
        [ ! -f $dir/feats_orin.scp ] &&
            mv $dir/feats.scp $dir/feats_orin.scp
        ln -snf feats_orin.scp $dir/feats.scp
    fi

done
rm -rf $fbank_conf
cd - >/dev/null

echo "$0 done."
exit 0
