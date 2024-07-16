#!/bin/bash

# Copyright 2020 Tsinghua SPMI Lab 
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com)
#
# Description:
#   This script saves raw audio files into Kaldi ark format for experimental purposes. It requires running local/data_multi.sh 
#   to obtain wav.scp and text files before executing this script. The script handles options for resampling and normalization 
#   of audio waveforms. The key steps include parsing command-line arguments, processing datasets, and packing audio files into 
#   Kaldi format.

set -e -u

<<"PARSER"
("datasets", type=str, nargs='+', help="Dataset(s) to be processed.")
("--resampling", type=int, default=0,help="Resampling rate. Default: none.")
("--normalize", type=int, default=1,help="Normalization waveform to [0., 1]. Default: 1, meaning use normalize.")
PARSER
eval $(python utils/parseopt.py $0 $*)

#exit 0
src="./data/src"

if [ $resampling -eq 0 ]; then
    opt_resampling=""
else
    opt_resampling="--resample $resampling"
    echo resample to $resampling
fi

if  [ $normalize -eq 1 ]; then
    opt_normalize=""
    ori=""
else
    opt_normalize="--skip-normalize"
    ori="ori"
    echo Skip Normalize
fi

#exit 0
for set in $datasets; do
    [ ! -d $src/$set ] && {
        echo "'$src/$set' not found." >&2
        exit 1
    }
    d_data=$src/${set}_raw_${ori}
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

    echo "python utils/tool/pack_audios_multi.py $opt_resampling $opt_normalize $f_wav ark,scp:$d_data/audio.ark,$d_data/feats.scp "
    #exit 0
    python utils/tool/pack_audios_multi.py \
        $opt_resampling $opt_normalize $f_wav \
        ark,scp:$d_data/audio.ark,$d_data/feats.scp ||
        exit 1
done

python utils/data/resolvedata.py

echo "$0 done." && exit 0
