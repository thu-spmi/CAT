"""
Compute FBank feature for yesno example using torchaudio.
"""

import os
import glob
import math
import argparse
from typing import List, Dict
from cat.utils.data.data_prep import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yesnodir", type=str, help="Directory to yesno audio files.")
    args = parser.parse_args()

    audios = glob.glob(f"{args.yesnodir}/?_?_?_?_?_?_?_?.wav")
    if len(audios) != 60:
        print(
            f"warning: there expect to be 60 audio files, instead found {len(audios)}"
        )
        fmtuid = r"yesno-{:0" + math.ceil(math.log10(len(audios))) + r"}"
    else:
        fmtuid = r"yesno-{:02}"

    odir = "data/src/yesno"
    os.makedirs(odir, exist_ok=True)

    mapping = {"0": "NO", "1": "YES"}
    with open(os.path.join(odir, "text"), "w") as f_text, open(
        os.path.join(odir, "wav.scp"), "w"
    ) as f_wav:
        for i, path in enumerate(audios):
            uid = fmtuid.format(i)
            trans = " ".join(
                mapping[x]
                for x in os.path.basename(path).split(".", maxsplit=1)[0].split("_")
            )
            f_text.write(f"{uid}\t{trans}\n")
            f_wav.write(f"{uid}\t{path}\n")

    prepare_kaldi_feat(
        subsets=["yesno"],
        trans=[os.path.join(odir, "text")],
        audios=[os.path.join(odir, "wav.scp")],
        read_from_extracted_meta=True,
        read_raw_data=True,
    )
