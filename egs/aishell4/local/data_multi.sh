#!/bin/bash

# Copyright 2020 Tsinghua SPMI Lab 
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com)
#
# Description:
#   This script prepares AISHELL data using torchaudio. It handles tasks such as checking for required files, 
#   running data preparation scripts, extracting FBank features, and removing spaces from text files. 
#   The key steps include parsing command-line arguments, preparing data subsets, and performing optional feature extraction.

set -e -u

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

<<"PARSER"
("-src", type=str, default="./data/src",
    help="Source data folder containing the text and wav.scp. ")
("-sp", type=float, nargs='*', default=None,
    help="Speed perturbation factor(s). Default: None.")
("-subsets", type=str, nargs="+", choices=["train", "dev", "test"],
    default=["train", "dev", "test"], help="Subset(s) for extracting FBanks. Default: ['train', 'dev', 'test']")
("-datapath", type=str, default=None,
    help="Aishell4 data path.Ensure that the path contains the original train and test")
("-fbank", type=bool, default=False,
    help="Calculate fbank or not")
PARSER
eval $(python utils/parseopt.py $0 $*)

opt_sp="1.0"
[ "$sp" != "None" ] && export opt_sp=$sp

# Create the source data folder if it doesn't exist
mkdir -p $src

for subset in ${subsets[@]}; do
    subset_folder="$src/$subset"
    
    # Create subset folder if it doesn't exist
    mkdir -p $subset_folder
    
    text_file="$subset_folder/text"
    wavscp_file="$subset_folder/wav.scp"
    
    # Check if text and wav.scp files exist, and run ori_data_prep.py if not
    if [ ! -f "$text_file" ] || [ ! -f "$wavscp_file" ]; then
        echo "dealing with AISHELL4, waiting......"
        python local/ori_data_prep.py $datapath
    fi
done

echo "File existence check completed!"
# exit 0
if [ "$fbank" = true ]; then
    # Extract 80-dim FBank features
    python local/extract_fbank_multi.py $src \
        --subset $subsets --speed-perturbation $opt_sp || exit 1

    echo "Extract 80-dim FBank features completed!"

    python utils/data/resolvedata.py
fi

# remove spaces
python -c "
import sys,os,shutil
sys.path.append('.')
import utils.pipeline.common_utils as cu
for f in cu.readjson('data/metainfo.json').values():
    f = f['trans']
    if os.path.isfile(f+'.bak'):
        continue
    dst = list(cu.get_corpus(adding_data=[f], ops=['rm-space'], skipid=True))[0]
    shutil.move(f, f+'.bak')
    shutil.move(dst, f)
"

echo "$0 done"
exit 0
