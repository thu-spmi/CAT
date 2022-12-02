#!/bin/bash
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# This script prepare aishell data by torchaudio
set -e -u

[ ! $(command -v python) ] && (
    echo "No python executable found in PATH"
    exit 1
)

<<"PARSER"
("-src", type=str, default="/data/aishell",
    help="Source data folder containing the audios and transcripts. "
        "Download from https://www.openslr.org/resources/33/data_aishell.tgz")
("-sp", type=float, nargs='*', default=None,
    help="Speed perturbation factor(s). Default: None.")
("-subsets-fbank", type=str, nargs="+", choices=["train", "dev", "test"],
    default=["train", "dev", "test"], help="Subset(s) for extracting FBanks. Default: ['train', 'dev', 'test']")
PARSER
eval $(python utils/parseopt.py $0 $*)

opt_sp="1.0"
[ "$sp" != "None" ] && export opt_sp=$sp

# Extract 80-dim FBank features
python local/extract_meta.py $src/wav \
    $src/transcript/aishell_transcript_v0.8.txt \
    --subset $subsets_fbank --speed-perturbation $opt_sp || exit 1

python utils/data/resolvedata.py

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
