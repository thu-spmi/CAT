"""
Interpolate multiple nbest list files to get the final scores
"""

import os
import pickle
import argparse


def nbest_add_ins_penalty_(nbest: dict, tip: float, tokenizer) -> dict:
    """In-place add token insert penalty to the nbest list"""
    for uid, hypos in nbest.items():
        for nid, (score, trans) in hypos.items():
            # "1+" because the </s> symbol
            hypos[nid] = (score+tip*(1+len(tokenizer.encode(trans))), trans)
    return nbest


def nbest_rescale_(nbest: dict, scale: float) -> dict:
    """In-place re-scale the nbest list with give scale factor."""
    for uid, hypos in nbest.items():
        for nid, (score, trans) in hypos.items():
            hypos[nid] = (score*scale, trans)
    return nbest


def nbest_add_with_scale_(nbestA: dict, nbestB: dict, scale: float) -> dict:
    """return nbestA + scale * nbestB, in-place modification of nbestA"""
    for uid, hypos in nbestA.items():
        for nid, (score, trans) in hypos.items():
            scoreB = nbestB[uid][nid][0]
            hypos[nid] = (score + scoreB * scale, trans)
    return nbestA


def main(args: argparse.Namespace):
    assert args.output is not None
    if args.ins_penalty != 0.0:
        assert args.tokenizer is not None, "--ins-penalty != 0.0 should be used with --tokenizer"
        import sys
        sys.path.append(os.getcwd())
        from cat.shared import tokenizer as tknz
        tokenizer = tknz.load(args.tokenizer)
        add_tip = True
    else:
        add_tip = False

    assert len(args.nbestlist) == len(args.weights), (
        f"\n--nbestlist and --weights should have the same number of arguments, however\n"
        f"# {len(args.nbestlist)}  --nbestlist {' '.join([str(x) for x in args.nbestlist])}\n"
        f"# {len(args.weights)}  --weights {' '.join([str(x) for x in args.weights])}")

    # see cat.share.data.NbestListDataset for the nbestlist file data format
    dest_nbest = {}
    for f_nbest, weight in zip(args.nbestlist, args.weights):
        if not os.path.isfile(f_nbest):
            raise FileNotFoundError(f"'{f_nbest}' is not a file.")

        with open(f_nbest, 'rb') as fi:
            part_nbest = pickle.load(fi)
        if len(dest_nbest) == 0:
            dest_nbest = nbest_rescale_(part_nbest, weight)
            if add_tip:
                dest_nbest = nbest_add_ins_penalty_(
                    dest_nbest, args.ins_penalty, tokenizer)
        elif weight != 0.0:
            dest_nbest = nbest_add_with_scale_(dest_nbest, part_nbest, weight)

    if args.one_best:
        one_best = {}
        with open(args.output, 'w') as fo:
            for uid, hypos in dest_nbest.items():
                best_hypo = max(hypos.values(), key=lambda item: item[0])[1]
                fo.write(f"{uid} {best_hypo}\n")
    else:
        with open(args.output, 'wb') as fo:
            pickle.dump(dest_nbest, fo)


def GetParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="Path to the output file.")
    parser.add_argument("--nbestlist", nargs='+', type=str,
                        help="N-best list files.")
    parser.add_argument("--weights", nargs='+', type=float, default=[1.0],
                        help="Weights for the n-best list files. Should be of equal number.")
    parser.add_argument("--one-best", action="store_true",
                        help="Output the 1-best path of the nbest list, instead the whole nbest list.")
    parser.add_argument("--ins-penalty", type=float, default=0.0,
                        help="The token insert penalty factor, should be used with --tokenizer. default: 0.0")
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer for inserting penalty.")
    return parser


if __name__ == "__main__":
    parser = GetParser()
    args = parser.parse_args()
    main(args)
