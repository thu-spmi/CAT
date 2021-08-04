"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Huahuan Zheng (zhh20@mails.tsinghua.edu.cn)
"""

import os
import argparse
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--prev_tr", type=str, default=None,
                                help="Original training set.")
    parser.add_argument("--prev_dev", type=str, default=None,
                                help="Original dev set.")
    parser.add_argument("--to_tr", type=str, default=None,
                                help="Output train set file in .tsv.")
    parser.add_argument("--to_dev", type=str, default=None,
                                help="Output dev set file in .tsv.")

    args = parser.parse_args()
    np.random.seed(0)

    for inarg in [args.prev_tr, args.prev_dev]:
        assert inarg is not None and os.path.isfile(
                inarg), f"File not found: {inarg}."

    origin_dev = pd.read_csv(args.prev_dev, sep="\t")
    length_dev = len(origin_dev)

    origin_tr = pd.read_csv(args.prev_tr, sep="\t")

    merged = pd.concat([origin_tr, origin_dev], ignore_index=True)

    # shuffle the order
    merged = merged.sample(frac=1).reset_index(drop=True)

    new_dev = merged[:length_dev]
    new_tr = merged[length_dev:]

    new_dev.to_csv(args.to_dev, sep='\t', index=False)
    new_tr.to_csv(args.to_tr, sep='\t', index=False)

