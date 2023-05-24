# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""Convert text to pickled data

text could be in token index format or pure text (with tokenizer specified)

would create two files: (suppose --output=<f_text>)
    <f_text> and <f_text>.bin
    where <f_text> stores the location of <f_text>.bin, as well as 
    the data location given by fseek(). 

How to get the parsed data:
    with open('<f_text>', 'rb') as fi:
        # mode: 0 for 1 sentence per item, 1 for tuple of two sentence (input, target)
        f_bin = os.path.join(os.path.dirname('<f_text>'), pickle.load(fi))
        f_seeks = pickle.load(fi)
    
    with open(f_bin, 'rb') as fi:
        # get the 5th element
        index = 5
        fi.seek(f_seeks[index], 0)
        data = pickle.load(fi)
"""

from cat.shared import tokenizer as tknz

import argparse
import pickle
import os
import uuid
import numpy as np
from multiprocessing import Pool
from typing import *


def chunk(chunk_size: int):
    def _chunk(
        X: Iterable[List[int]],
    ) -> Iterator[List[int]]:
        for x in X:
            for bound in range(0, len(x), chunk_size):
                yield x[bound : bound + chunk_size]
        return

    return _chunk


def text2bin(arguments: Tuple[argparse.Namespace, str, int, int]):
    args, binfile, idx_beg, idx_end = arguments
    if idx_end == -1:
        idx_end = float("inf")

    tokenizer = tknz.load(args.tokenizer)
    processor = tokenizer.encode
    if args.skip_control_sym:
        bos, eos = [], []
    else:
        if args.bos_id == -1:
            bos = tokenizer.get_bos_id()
        else:
            bos = args.bos_id

        if args.eos_id == -1:
            eos = tokenizer.get_eos_id()
        else:
            eos = args.eos_id

        if eos == -1:
            eos = bos

        bos, eos = [bos], [eos]

    def file_reader() -> Iterator[List[int]]:
        with open(args.intext, "r") as fi:
            for i, line in enumerate(fi):
                if i < idx_beg:
                    continue
                if i >= idx_end:
                    break
                yield bos + processor(line.strip()) + eos
        return

    if args.truncate == -1:
        reader = file_reader()
    else:
        reader = chunk(args.truncate)(file_reader())

    _seeks = []
    _lens = []
    with open(binfile, "wb") as fo:
        for indices in reader:
            if len(indices) < args.prune_shorter:
                continue
            _seeks.append(fo.tell())
            pickle.dump(indices, fo)
            _lens.append(len(indices))

    return (
        binfile,
        np.asarray(_seeks, dtype=np.int64),
        np.asarray(_lens, dtype=np.int64),
    )


def main(args: argparse.Namespace):
    num_threads = args.nj
    if num_threads < 1:
        raise ValueError(f"#threads must be >= 1, instead: {num_threads}")

    if not os.path.isfile(args.intext):
        raise FileNotFoundError(f"{args.intext} does not exist!")

    fmt = args.output + ".{}.bin"
    if num_threads == 1:
        pool_args = [(args, fmt.format(0), 0, -1)]
    else:
        num_lines = sum(1 for _ in open(args.intext, "r"))
        interval = num_lines // num_threads
        indices = [interval * i for i in range(num_threads + 1)]
        if indices[-1] != num_lines:
            indices[-1] = num_lines

        pool_args = [
            (args, fmt.format(i), indices[i], indices[i + 1])
            for i in range(num_threads)
        ]

    with Pool(processes=num_threads) as pool:
        collect_seeks = pool.map(
            text2bin, pool_args
        )  # type: List[Tuple[str, np.ndarray, np.ndarray]]

    with open(args.output, "wb") as fo:
        # save the file name of binary file
        pickle.dump([os.path.basename(x) for x, _, _ in collect_seeks], fo)
        # save the seq length information
        pickle.dump(np.concatenate([l for _, _, l in collect_seeks], axis=0), fo)
        # save the file seeking information
        pickle.dump([s for _, s, _ in collect_seeks], fo)

    if not args.quiet:
        print("> Corpus packed: {}".format(args.output))


def _parser():
    parser = argparse.ArgumentParser(
        "Convert pure text into pickle data with multi-processing"
    )
    parser.add_argument("intext", type=str, help="Input text files.")
    parser.add_argument("output", type=str, help="Ouput file.")
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer model file. See cat/shared/tokenizer.py for details.",
    )
    parser.add_argument(
        "--nj", type=int, default=8, help="Number of threads. Default: 8"
    )
    parser.add_argument(
        "--skip-control-sym",
        action="store_true",
        help="Disable adding extra <s> and </s> to data.",
    )
    parser.add_argument(
        "--prune-shorter",
        type=int,
        default=-1,
        help="Eliminate sequences shorter than given length (after padding <s> and </s>). Default: -1 (disable)",
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=-1,
        metavar="trunc",
        help="Truncate the seq longer than trunc size (after padding <s> and </s>) and take res of it as new seq. Default: -1 (disable)",
    )
    parser.add_argument(
        "--bos_id",
        type=int,
        default=-1,
        help="Index of control symbol: <s> (begin of sequences). Default: -1 (get from tokenizer)",
    )
    parser.add_argument(
        "--eos_id",
        type=int,
        default=-1,
        help="Index of control symbol: </s> (end of sequences). Default: -1 (see --bos_id)",
    )
    parser.add_argument("--quiet", action="store_true", help="Supress output messages.")
    return parser


if __name__ == "__main__":
    parser = _parser()
    args = parser.parse_args()
    main(args)
