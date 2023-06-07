#!/bin/bash
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# This script prepare commonvoice data by torchaudio
# ... the wav.scp file is not in standard kaldi format.
set -e -u

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

<<"PARSER"
("src", type=str, help="Source data folder containing the audios and transcripts.")
("-sp", type=float, nargs='*', default=None,
    help="Speed perturbation factor(s). Default: None.")
("-lang", type=str, default='', help="Identifier for language, e.g., en, zh, ...")
PARSER
eval $(python utils/parseopt.py $0 $*)

dst_dir="data/src"

opt_sp="1.0"
[ "$sp" != "None" ] && export opt_sp=$sp

[ ! -z $lang ] && export lang="$lang-"

# Extract meta info
mkdir -p $dst_dir
for s in dev test train validated; do
    d_set="$dst_dir/$lang$s"
    mkdir -p $d_set
    file="$src/$s.tsv"
    [ ! -f $file ] && {
        echo "No such file $file"
        exit 1
    }
    cut <$file -f 2 | tail -n +2 | xargs basename -s ".mp3" >$d_set/uid.tmp
    cut <$file -f 2 | tail -n +2 | awk -v path="$src/clips" '{print path"/"$1}' >$d_set/path.tmp
    paste $d_set/{uid,path}.tmp | sort -k 1,1 -u >$d_set/wav.scp
    cut <$file -f 3 | tail -n +2 >$d_set/text.tmp
    paste $d_set/{uid,text}.tmp | sort -k 1,1 -u >$d_set/text
    rm -rf $d_set/{uid,text,path}.tmp
done

# By default, I use validated+train as the real training data
# ... but we must exclude the dev & test from the validated one.
d_train="$dst_dir/${lang}excluded_train"
mkdir -p $d_train
for file in wav.scp text; do
    cat $dst_dir/${lang}{validated,train}/$file |
        sort -k 1,1 -u >$d_train/$file.tmp
    for exc_set in dev test; do
        python utils/data/exclude_corpus.py \
            $d_train/$file.tmp \
            --exclude $dst_dir/$lang$exc_set/$file \
            >$d_train/$file.tmp.tmp
        mv $d_train/$file.tmp.tmp $d_train/$file.tmp
    done
    mv $d_train/$file.tmp $d_train/$file
done

# Extract 80-dim FBank features
python local/make_fbank.py $dst_dir \
    --subset $(printf "$lang%s " dev test excluded_train) \
    --speed-perturbation $opt_sp ||
    exit 1

python utils/data/resolvedata.py

# Do some simple text-normalize
bash local/text_normalize.sh

echo "$0 done"
exit 0
