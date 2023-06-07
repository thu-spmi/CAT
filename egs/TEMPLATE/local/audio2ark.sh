#!/bin/bash
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# This script directly save raw audios into kaldi ark (for experiment reading raw audios)
# Run local/data.sh to obtain wav.scp and text before this script.

set -e -u

<<"PARSER"
("datasets", type=str, nargs='+', help="Dataset(s) to be processed.")
("--resampling", type=int, default=0,
    help="Resampling rate. Default: none.")
PARSER
eval $(python utils/parseopt.py $0 $*)

src="data/src"

if [ $resampling -eq 0 ]; then
    opt_resampling=""
else
    opt_resampling="--resample $resampling"
fi

for set in $datasets; do
    [ ! -d $src/$set ] && {
        echo "'$src/$set' not found." >&2
        exit 1
    }
    d_data=$src/${set}_raw
    mkdir -p $d_data
    cp $src/$set/text $d_data
    [[ -f $d_data/feats.scp && $(wc -l <$d_data/feats.scp) -eq $(wc -l <$d_data/text) ]] && {
        echo "$set: existing feats.scp found, skip" >&2
        continue
    }

    f_wav="$src/$set/wav.scp"
    # check format of wav.scp
    suffix=$(head -n 1 $src/$set/wav.scp | grep -o '[^.]*$')
    if [ "$suffix" != "wav" ]; then
        # check whether there is a pipe out
        last_char=$(head -n 1 $src/$set/wav.scp | grep -o '.\s*$')
        if [ "$last_char" != "|" ]; then
            awk <$src/$set/wav.scp '{print $1,"ffmpeg -v 0 -i",$2,"-f wav - |"}' >$d_data/wav.tmp
            f_wav="$d_data/wav.tmp"
        fi
    fi

    python utils/tool/pack_audios.py \
        $opt_resampling $f_wav \
        ark,scp:$d_data/audio.ark,$d_data/feats.scp ||
        exit 1
done

python utils/data/resolvedata.py

echo "$0 done." && exit 0
