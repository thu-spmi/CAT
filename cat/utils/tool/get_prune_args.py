"""
Author: Huahuan Zheng

Get prune argument for kenlm lmplz

e.g.
    lmplz --prune 0 1 2 -o 3 <in.txt >out.arpa

0 means not prune 1-gram;
1 means pruning the 2-grams occurring less than 1 time;
2 means pruning the 3-grams occurring less than 2 times.

However, sometimes we want to prune according to the top-k
    most freqeuent n-grams (like LODR method).
This script is designed for translating the top-k most frequent
    n-grams to kenlm --prune parameters (not 100% exact)

Return to stdout:
    (prune-arg, real-topk)
"""

import os
import argparse
import struct
from typing import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ngram_counts", type=str,
                        help="File of n-gram counts, see https://github.com/kpu/kenlm/issues/201 and https://github.com/kpu/kenlm/issues/378#issuecomment-1085156101")
    parser.add_argument("order", type=int, help="Order of n-gram counts.")
    parser.add_argument("topk", type=int, help="Top-k most frequent n-grams.")
    parser.add_argument("--keep_l_bound", action="store_true",
                        help="When the translation is not strictly conduct, "
                        "keep lower bound. Defaultly keep the upper bound (more n-grams than <topk>).")
    args = parser.parse_args()

    assert os.path.isfile(args.ngram_counts), args.ngram_counts

    order = int(args.order)
    assert order >= 1
    topk = int(args.topk)
    assert topk >= 1

    anno = '=' + 'I'*order + 'Q'
    ngram_count = []    # type: List[int]
    with open(args.ngram_counts, 'rb') as fib:
        while record := fib.read(order * 4 + 8):
            value = struct.unpack(anno, record)
            # ngram = value[:order]
            count = value[-1]
            ngram_count.append(count)

    ngram_count = sorted(ngram_count, reverse=True)
    if topk >= len(ngram_count):
        print(0, len(ngram_count))
        exit(0)

    trial = ngram_count[topk-1]
    if args.keep_l_bound:
        if ngram_count[0] == trial:
            print(ngram_count[0], 1)
            exit(0)
        direc = -1
    else:
        if ngram_count[-1] == trial:
            print(0, len(ngram_count))
            exit(0)
        direc = 1

    while ngram_count[topk-1] == trial:
        topk += direc

    if args.keep_l_bound:
        print(ngram_count[topk-1], topk)
    else:
        print(ngram_count[topk-2], topk-1)
