"""
Resolve the data location from data/src.

e.g.
$ cd path_to_transducer/egs/wsj
$ python utils/data/resolvedata.py

Find datasets in `data/src`, which should satisfy:
    1. Foler `data/src/SET` exist
    2. Files `data/src/SET/feats.scp` and `data/src/SET/text` exist

The found datasets info would be stored at `data/metainfo.json` in JSON format
    {
        "SET":{
            "scp": "/abs/path/to/data/src/SET/feats.scp",
            "trans" : "/abs/path/to/data/src/SET/text"
        },
        ...
    }

All following pipeline of model training would depend on `data/metainfo.json`.
You can modify the file manually for more flexible usage.
"""

import os
import sys
import json
from typing import Dict, List
# fmt:off
try:
    import utils.pipeline
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../..'))
from utils.pipeline._constants import *
# fmt:on

D_SRCDATA = 'data/src'


def find_dataset(d_data: str) -> Dict[str, Dict[str, str]]:
    assert os.path.isdir(d_data), f"'{d_data}' is not a directory."

    datasets = {}
    for d in os.listdir(d_data):
        _setdir = os.path.join(d_data, d)
        if not os.path.isdir(_setdir):
            continue

        check_cnt = 0
        for f in os.listdir(_setdir):
            if f == 'feats.scp':
                check_cnt += 1
            elif f == 'text':
                check_cnt += 1

            if check_cnt == 2:
                break
        if check_cnt != 2:
            continue

        datasets[d] = {
            'scp': os.path.abspath(os.path.join(_setdir, 'feats.scp')),
            'trans': os.path.abspath(os.path.join(_setdir, 'text')),
        }
    return datasets


def main():
    if os.path.isdir(D_SRCDATA):
        found_datasets = find_dataset(D_SRCDATA)
    else:
        found_datasets = {}
        sys.stderr.write(
            f"speech data resolve: {D_SRCDATA} is not found, did you run the pre-processing steps?\n")

    if os.path.isfile(F_DATAINFO):
        backup = json.load(open(F_DATAINFO, 'r'))
        meta = backup.copy()
    else:
        os.makedirs(os.path.dirname(F_DATAINFO), exist_ok=True)
        backup = None
        meta = {}

    meta.update(found_datasets)
    try:
        with open(F_DATAINFO, 'w') as fo:
            json.dump(meta, fo, indent=4, sort_keys=True)
    except Exception as e:
        if backup is not None:
            with open(F_DATAINFO, 'w') as fo:
                json.dump(backup, fo, indent=4, sort_keys=True)
        raise RuntimeError(str(e))


if __name__ == "__main__":
    main()
