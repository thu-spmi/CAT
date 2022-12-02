# Copyright 2021 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""Rescoring with custom LM.

This script is ported from cat.rnnt.decode

Rescore N-best list.

P.S. CPU is faster when rescoring with n-gram model, while GPU
     would be faster rescoring with NN model.
"""

from . import lm_builder
from ..shared import tokenizer as tknz
from ..shared import coreutils
from ..shared.decoder import (
    AbsDecoder,
    NGram
)
from ..shared.data import (
    NbestListDataset,
    NbestListCollate
)

import os
import sys
import time
import pickle
import argparse
from tqdm import tqdm
from typing import *

import torch
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    assert os.path.isfile(
        args.nbestlist), f"N-best list file not found: {args.nbestlist}"
    assert args.tokenizer is not None, "You need to specify --tokenizer."
    assert os.path.isfile(
        args.tokenizer), f"Tokenizer model not found: {args.tokenizer}"

    if args.cpu or not torch.cuda.is_available():
        args.cpu = True

    if args.cpu:
        if args.nj == -1:
            world_size = max(os.cpu_count()//2, 1)
        else:
            world_size = args.nj
    else:
        world_size = torch.cuda.device_count()
    args.world_size = world_size

    try:
        mp.set_start_method('spawn')
    except RuntimeError as re:
        if args.verbose:
            print(re)
    q_data = mp.Queue(maxsize=1)
    producer = mp.Process(target=dataserver, args=(args, q_data))
    producer.start()

    q_out = mp.Queue(maxsize=1)
    consumer = mp.Process(target=datawriter, args=(args, q_out))
    consumer.start()

    if args.cpu:
        model = build_lm(args.config, args.resume, 'cpu')
        model.share_memory()
        mp.spawn(main_worker, nprocs=world_size,
                 args=(args, q_data, q_out, model))
    else:
        mp.spawn(main_worker, nprocs=world_size,
                 args=(args, q_data, q_out))
    producer.join()
    consumer.join()

    del q_data
    del q_out


def dataserver(args, q: mp.Queue):
    testset = NbestListDataset(args.nbestlist)
    tokenizer = tknz.load(args.tokenizer)
    testloader = DataLoader(
        testset, batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=NbestListCollate(tokenizer))

    t_beg = time.time()
    for batch in tqdm(testloader, desc="LM rescore", total=len(testloader), disable=(not args.verbose), leave=False):
        for k in batch:
            if isinstance(k, torch.Tensor):
                k.share_memory_()
        q.put(batch, block=True)

    # put one more None, so that it guarantees
    # ... all workers get the exit flag.
    for _ in range(args.world_size+1):
        q.put(None, block=True)

    if args.verbose:
        print("Time = {:.2f} s".format(time.time() - t_beg))


def datawriter(args, q: mp.Queue):
    nbest = {}
    cnt_done = 0
    save_also_nbest = (args.save_lm_nbest is not None)

    with open(args.output, 'w') as fo:
        while True:
            data = q.get(block=True)
            if data is None:
                cnt_done += 1
                if cnt_done == args.world_size:
                    break
                continue

            if save_also_nbest:
                nbest.update(data[1])
            for k, (_, t) in data[0].items():
                fo.write(f"{k}\t{t}\n")
            del data

    if save_also_nbest:
        os.makedirs(os.path.dirname(args.save_lm_nbest), exist_ok=True)
        with open(args.save_lm_nbest, 'wb') as fo:
            pickle.dump(nbest, fo)


def main_worker(pid: int, args: argparse.Namespace, q_data: mp.Queue, q_out: mp.Queue, model=None):

    args.pid = pid
    args.rank = pid

    if args.cpu:
        device = 'cpu'
        torch.set_num_threads(1)
    else:
        device = pid
        torch.cuda.set_device(device)

    if model is None:
        model = build_lm(args.config, args.resume, device)

    lm_nbest = {}   # type: Dict[str, Dict[int, Tuple[float, str]]]
    # rescoring
    with torch.no_grad(),\
            autocast(enabled=(True if device != 'cpu' else False)):
        while True:
            batch = q_data.get(block=True)
            if batch is None:
                break
            keys, texts, scores, in_toks, mask = batch
            in_toks = in_toks.to(device)

            # suppose </s> = <s>
            dummy_targets = torch.roll(in_toks, -1, dims=1)
            in_lens = in_toks.size(1) - mask.sum(dim=1)
            log_lm_probs = model.score(in_toks, dummy_targets, in_lens)

            final_score = scores + args.alpha * log_lm_probs.cpu() + args.beta * in_lens
            indiv = {}
            for k, t, s in zip(keys, texts, final_score):
                _, okey = k.split('-', maxsplit=1)
                if okey not in indiv:
                    indiv[okey] = (s, t)
                elif indiv[okey][0] < s:
                    indiv[okey] = (s, t)

            if args.save_lm_nbest is not None:
                for _key, _trans, _score in zip(keys, texts, log_lm_probs):
                    nid, okey = _key.split('-', maxsplit=1)
                    if okey not in lm_nbest:
                        lm_nbest[okey] = {}
                    lm_nbest[okey][int(nid)] = (_score.item(), _trans)

                q_out.put((indiv, lm_nbest), block=True)
                lm_nbest = {}
            else:
                q_out.put((indiv, None), block=True)

            del batch

    time.sleep(0.5)
    q_out.put(None, block=True)


def build_lm(f_config: str, f_check: str, device='cuda') -> AbsDecoder:
    if isinstance(device, int):
        device = f'cuda:{device}'

    model = lm_builder(coreutils.readjson(f_config), dist=False)
    if f_check is None:
        sys.stderr.write(
            "WARNING: checkpoint is not configured. This MIGHT be OK if the LM is N-gram liked.\n")
    else:
        model = coreutils.load_checkpoint(model.to(device), f_check)
    model = model.lm
    model.eval()
    return model


def _parser():
    parser = coreutils.basic_trainer_parser(
        prog="Rescore with give n-best list and LM",
        training=False,
        isddp=False
    )

    parser.add_argument("nbestlist", type=str,
                        help="Path to N-best list files.")
    parser.add_argument("output", type=str, help="The output text file. ")

    parser.add_argument("--alpha", type=float, default=0.0,
                        help="The 'alpha' value for LM integration, a.k.a. the LM weight")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="The 'beta' value for LM integration, a.k.a. the penalty of tokens.")
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer model file. See cat/shared/tokenizer.py for details.")
    parser.add_argument("--save-lm-nbest", type=str,
                        help="Path to save the LM N-best scores.")

    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--verbose", action='store_true', default=False)

    return parser


if __name__ == "__main__":
    main()
