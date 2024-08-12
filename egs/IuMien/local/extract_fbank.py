import os
import glob
import argparse
from typing import List, Dict, Any, Tuple

# fmt: off
import sys
try:
    import utils.data
except ModuleNotFoundError:
    sys.path.append(".")
from utils.data import data_prep
# fmt: on


prepare_sets = [
    'train',
    'dev',
    'test'
]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_data", type=str, default="/home/dlk/code/asr/data/MightLJSpeech/MightLJSpeech-1.1",
                        help="Directory to source audio files, "
                        f"expect subset: {', '.join(prepare_sets)} in the directory.")
    parser.add_argument("--speed-perturbation", type=float, dest='sp',
                        nargs='*', default=[], help=f"Add speed perturbation to subset: {', '.join(prepare_sets)}")
    args = parser.parse_args()

    audios = {}     # type: Dict[str, List[Tuple[str, str]]]
    subtrans = {}   # type: Dict[str, List[Tuple[str, str]]]
    train_sets = {}
    dev_sets = {}
    test_sets = {}
    with open(args.src_data+'/dev_data.txt','r') as fi:
        for line in fi:
            audio_id,audio_trans = line.strip().split('\t')
            dev_sets[audio_id] = audio_trans
    with open(args.src_data+'/test_data.txt','r') as fi:
        for line in fi:
            audio_id,audio_trans = line.strip().split('\t')
            test_sets[audio_id] = audio_trans
    with open(args.src_data+'/train_data.txt','r') as fi:
        for line in fi:
            audio_id,audio_trans = line.strip().split('\t')
            train_sets[audio_id] = audio_trans
    _audios = glob.glob(f"{args.src_data}/wavs/*.wav")
    audios['train'] = []
    subtrans['train'] = []
    audios['dev'] = []
    subtrans['dev'] = []
    audios['test'] = []
    subtrans['test'] = []
    for _raw_wav in _audios:
        uid = os.path.basename(_raw_wav).removesuffix('.wav')
        if uid in dev_sets:
            audios['dev'].append((uid,_raw_wav))
            subtrans['dev'].append((uid,dev_sets[uid]))
        if uid in test_sets:
            audios['test'].append((uid,_raw_wav))
            subtrans['test'].append((uid,test_sets[uid]))
        if uid in train_sets:
            audios['train'].append((uid,_raw_wav))
            subtrans['train'].append((uid,train_sets[uid]))
    
    # print(audios['train'][:10])
    # print(subtrans['train'][:10])
    data_prep.prepare_kaldi_feat(
        subsets=prepare_sets,
        trans=subtrans,
        audios=audios,
        num_mel_bins=80,
        apply_cmvn=False,
        speed_perturb=args.sp
    )
