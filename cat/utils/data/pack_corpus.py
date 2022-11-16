"""Convert text to pickle data

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
from typing import Union, Tuple, List


def chunk(X: List[int],  chunk_size: int, drop_res: bool = True, Y: List[int] = None):
    lx = len(X)
    if drop_res:
        assert lx >= chunk_size
        res_size = lx % chunk_size
    else:
        res_size = 0

    if Y is None:
        for bound in range(0, lx-res_size, chunk_size):
            yield X[bound:bound+chunk_size]
    else:
        for bound in range(0, lx-res_size, chunk_size):
            yield X[bound:bound+chunk_size], Y[bound:bound+chunk_size]


def text2bin(arguments: Tuple[argparse.Namespace, str, int, int]):
    args, binfile, idx_beg, idx_end = arguments
    if idx_end == -1:
        idx_end = float('inf')

    processor = tknz.load(args.tokenizer).encode
    bos = [args.bos_id]
    eos = [args.eos_id]

    def file_reader():
        with open(args.intext, 'r') as fi:
            for i, line in (enumerate(fi)):
                if i < idx_beg:
                    continue
                if i >= idx_end:
                    break
                yield bos + processor(line.strip())+eos
        return

    with open(binfile, 'wb') as fo:
        # mode = 0
        if args.truncate != -1:
            # mode = 1
            chunksize = args.truncate
            for indices in file_reader():
                for x, y in chunk(indices[:-1], chunksize, drop_res=False, Y=indices[1:]):
                    pickle.dump((x, y), fo)
        elif args.concat != -1:
            chunksize = args.concat
            for indices in file_reader():
                for x in chunk(indices, chunksize, drop_res=True):
                    pickle.dump(x, fo)
        else:
            for indices in file_reader():
                pickle.dump(indices, fo)
        # terminated flag
        pickle.dump(None, fo)

    return


def main(args: argparse.Namespace):
    assert args.truncate == -1 or args.concat == - \
        1, "--concat is conflict with --truncate"

    num_threads = args.nj
    if num_threads < 1:
        raise ValueError(f"# threads must be >= 1, instead: {num_threads}")

    if not os.path.isfile(args.intext):
        raise FileNotFoundError(f"{args.intext} does not exist!")

    if args.eos_id == -1:
        args.eos_id = args.bos_id

    if args.concat > 1 and (args.eos_id != args.bos_id):
        raise RuntimeError(
            f"--concat > 1 requires <bos> = <eos>, instead {args.bos_id} != {args.eos_id}")

    fmt = os.path.join('/tmp', str(uuid.uuid4())+'.{}.tmp')
    if num_threads == 1:
        pool_args = [(args, fmt.format(0), 0, -1)]
    else:
        num_lines = sum(1 for _ in open(args.intext, 'r'))
        interval = num_lines // num_threads
        indices = [interval * i for i in range(num_threads+1)]
        if indices[-1] != num_lines:
            indices[-1] = num_lines

        pool_args = [(args, fmt.format(i), indices[i], indices[i+1])
                     for i in range(num_threads)]

    with Pool(processes=num_threads) as pool:
        pool.map(text2bin, pool_args)

    if not args.quiet:
        print("> Sub-process done. Begin merging...")

    f_data = '{}.bin'.format(args.output)
    _seeks = []
    with open(f_data, 'wb') as fo:
        for i in range(num_threads):
            with open(fmt.format(i), 'rb') as fi:
                while (data := pickle.load(fi)) != None:
                    _seeks.append(fo.tell())
                    pickle.dump(data, fo)
            os.remove(fmt.format(i))

    with open(args.output, 'wb') as fo:
        # save the file name of binary file
        pickle.dump(os.path.basename(f_data), fo)
        # save the location information
        pickle.dump(np.asarray(_seeks, dtype=np.int64), fo)

    if not args.quiet:
        print("> Merged: Index {} --> binary {}".format(args.output, f_data))


def _parser():
    parser = argparse.ArgumentParser(
        'Convert pure text into pickle data with multi-processing')
    parser.add_argument("intext", type=str,
                        help="Input text files.")
    parser.add_argument("output", type=str, help="Ouput file.")
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer model file. See cat/shared/tokenizer.py for details.")
    parser.add_argument("--nj", type=int, default=1,
                        help="Number of threads. Default: 1")
    parser.add_argument("--concat", type=int, default=-1,
                        help="Use concat mode instead valid mode with given length. Default: -1 (disable)")
    parser.add_argument("--truncate", type=int, default=-1, metavar="trunc",
                        help="Truncate the seq longer than trunc and take res of it as new seq. Default: -1 (disable)")
    parser.add_argument("--bos_id", type=int, default=0,
                        help="Begin of sequence index, used when concat > 1. Default: 0")
    parser.add_argument("--eos_id", type=int, default=-1,
                        help="End of sequence index, used when concat > 1. Default: -1 (same as --bos_id)")
    parser.add_argument("--quiet", action="store_true",
                        help="Supress hint messages")
    return parser


if __name__ == "__main__":
    parser = _parser()
    args = parser.parse_args()
    main(args)
