"""
Fetch IPA subset matrix from IPA full-set.

source:
https://raw.githubusercontent.com/dmort27/panphon/master/panphon/data/ipa_all.csv
"""
import pandas as pd
import numpy as np
import pickle

import os
import sys
import argparse

from typing import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phonemes", type=str, help="Units of target IPA symbols.")
    parser.add_argument(
        "ipa_all",
        type=str,
        help=(
            "The .csv file contains the full IPA mappings, "
            "which can be found at https://raw.githubusercontent.com/dmort27/panphon/master/panphon/data/ipa_all.csv"
        ),
    )
    parser.add_argument("output", type=str, help="Output pickled file.")

    args = parser.parse_args()

    dst = []
    # add special tokens <blk> and <unk>
    st_blk = np.zeros((51,), dtype=np.int64)
    st_blk[48] = 1
    st_unk = np.zeros_like(st_blk)
    st_unk[49] = 1
    # NOTE: spn is not necessary, but only for matching the paper.
    # st_spn = np.zeros_like(st_blk)
    # st_spn[50] = 1
    # dst.extend([st_blk, st_unk, st_spn])
    dst.extend([st_blk, st_unk])

    # read the ipa all file
    ipa_syms = pd.read_csv(args.ipa_all)
    length = len(ipa_syms["ipa"])
    indexing = {ipa_syms["ipa"][i]: i for i in range(length)}  # type: Dict[str, int]

    # read symbols from the existing units
    exists = open(args.phonemes, "r").read().strip().split("\n")

    mapping = {
        "+": np.asarray([1, 0], dtype=np.int64),
        "-": np.asarray([0, 1], dtype=np.int64),
        "0": np.asarray([0, 0], dtype=np.int64),
    }

    notfound = []
    for sym in exists:
        if sym not in indexing:
            notfound.append(sym)
            continue

        i = indexing[sym]
        token = np.zeros_like(st_blk)
        for k, ipa in enumerate(ipa_syms.keys()):
            if k == 0:
                # 'ipa'
                continue
            token[(k - 1) * 2 : k * 2] = mapping[ipa_syms[ipa][i]]
        dst.append(token)

    dst = np.stack(dst)
    np.save(args.output, dst)

    if len(notfound) > 0:
        sys.stderr.write(
            f"WARNING: there is(are) {len(notfound)} symbol(s) not found in the IPA tables.\n"
            f"{notfound}\n"
        )
        sys.stderr.write(f"missing: {len(notfound)} | found: {dst.shape[0]}\n")
