# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""Convert n-best list data from raw texts into custom .nbest format.

used with cat/ctc/fst_decode.py
"""
import pickle
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trans", type=str, help="Transcripts of N-best list")
    parser.add_argument("ac_score", type=str, help="AC-score file of N-best list")
    parser.add_argument(
        "output", type=str, help="Output binary N-best file. Usually as *.nbest"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.trans):
        raise FileNotFoundError(f"No such transcript file '{args.trans}'")
    if not os.path.isfile(args.ac_score):
        raise FileNotFoundError(f"No such ac-score file '{args.ac_score}'")

    hypos = {}
    with open(args.trans, "r") as fi:
        for line in fi:
            uidhyp = line.strip().split(maxsplit=1)
            if len(uidhyp) == 1:
                hypos[uidhyp[0]] = ""
            else:
                hypos[uidhyp[0]] = uidhyp[1]

    ac_scores = {}
    with open(args.ac_score, "r") as fi:
        for line in fi:
            uid, _score = line.strip().split(maxsplit=1)
            ac_scores[uid] = -float(_score)
            if uid not in hypos:
                raise RuntimeError(
                    f"'{uid}' found in '{args.ac_score}' but not in '{args.trans}'"
                )

    if len(ac_scores) != len(hypos):
        raise RuntimeError(
            f"Number of utterances mismatch between two input files: {len(hypos)} != {len(ac_scores)}"
        )

    nbest = {}
    for uid in hypos:
        _, real_uid = uid[::-1].split("-", maxsplit=1)
        real_uid = real_uid[::-1]
        if real_uid not in nbest:
            nbest[real_uid] = {}
        nbest[real_uid][len(nbest[real_uid])] = (ac_scores[uid], hypos[uid])

    with open(args.output, "wb") as fo:
        pickle.dump(nbest, fo)
