# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""
Parallel decode with distributed GPU/CPU support 
"""

from .beam_search import BeamSearcher
from ..lm import lm_builder
from ..shared import coreutils
from ..shared import tokenizer as tknz
from ..shared.data import (
    ScpDataset,
    sortedScpPadCollate
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
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    if args.tokenizer is None or not os.path.isfile(args.tokenizer):
        raise FileNotFoundError(
            "Invalid tokenizer model file: {}".format(args.tokenizer))
    if args.cpu or not torch.cuda.is_available():
        args.cpu = True

    if args.cpu:
        if args.nj == -1:
            world_size = os.cpu_count()
        else:
            world_size = args.nj
    else:
        world_size = torch.cuda.device_count()
    args.world_size = world_size

    try:
        mp.set_start_method('spawn')
    except RuntimeError as re:
        print(re)

    q_data_producer = mp.Queue(maxsize=1)
    q_nbest_saver = mp.Queue(maxsize=1)
    producer = mp.Process(target=dataserver, args=(args, q_data_producer))
    consumer = mp.Process(target=datawriter, args=(args, q_nbest_saver))
    producer.start()
    consumer.start()

    if args.cpu:
        model, ext_lm = build_model(args, 'cpu')
        model.share_memory()
        if ext_lm is not None:
            ext_lm.share_memory()

        mp.spawn(main_worker, nprocs=world_size,
                 args=(args, q_data_producer, q_nbest_saver, (model, ext_lm)))
    else:
        mp.spawn(main_worker, nprocs=world_size,
                 args=(args, q_data_producer, q_nbest_saver))

    producer.join()
    consumer.join()
    del q_nbest_saver
    del q_data_producer


def dataserver(args, q: mp.Queue):
    testset = ScpDataset(args.input_scp)
    # sort the dataset in desencding order
    testset_ls = testset.get_seq_len()
    len_match = sorted(list(zip(testset_ls, testset._dataset)),
                       key=lambda item: item[0], reverse=True)
    testset._dataset = [data for _, data in len_match]
    n_frames = sum(testset_ls)
    del len_match, testset_ls
    testloader = DataLoader(
        testset,
        batch_size=max(1, min(8, len(testset)//args.world_size)),
        shuffle=False,
        num_workers=1,
        collate_fn=sortedScpPadCollate())

    f_nbest = args.output_prefix+'.nbest'
    if os.path.isfile(f_nbest):
        with open(f_nbest, 'rb') as fi:
            nbest = pickle.load(fi)
    else:
        nbest = {}

    t_beg = time.time()
    for batch in tqdm(testloader, desc="RNN-T decode", total=len(testloader), disable=(args.silent), leave=False):
        key = batch[0][0]
        """
        NOTE (Huahuan):
        In some cases (decoding with large beam size or large LMs like Transformer), 
        ... the decoding consumes too much memory and would probably causes OOM error.
        So I add checkpointing output in the nbest list. However, things would be
        ... complicated say if we first decoding w/o lm, save the checkpoint to nbest list
        ... then continue decoding w/ lm.
        I just assume users won't do that.
        """
        if key not in nbest:
            q.put(batch, block=True)

    for i in range(args.world_size+1):
        q.put(None, block=True)
    t_dur = time.time() - t_beg

    if not args.silent:
        print("Time = {:.2f} s | RTF = {:.2f} ".format(
            t_dur, t_dur*args.world_size / n_frames * 100))


def datawriter(args, q: mp.Queue):
    """Get data from queue and save to file."""
    def load_and_save(_nbest: dict):
        if os.path.isfile(f_nbest):
            with open(f_nbest, 'rb') as fi:
                _nbest.update(pickle.load(fi))
        with open(f_nbest, 'wb') as fo:
            pickle.dump(_nbest, fo)

    f_nbest = args.output_prefix+'.nbest'
    interval_check = 1000   # save nbestlist to file every 1000 steps
    cnt_done = 0
    nbest = {}
    while True:
        nbestlist = q.get(block=True)
        if nbestlist is None:
            cnt_done += 1
            if cnt_done == args.world_size:
                break
            continue
        nbest.update(nbestlist)
        del nbestlist
        if len(nbest) % interval_check == 0:
            load_and_save(nbest)
            nbest = {}

    load_and_save(nbest)
    # write the 1-best result to text file.
    with open(args.output_prefix, 'w') as fo:
        for k, hypo_items in nbest.items():
            best_hypo = max(hypo_items.values(), key=lambda item: item[0])[1]
            fo.write(f"{k}\t{best_hypo}\n")

    del load_and_save


def main_worker(pid: int, args: argparse.Namespace, q_data: mp.Queue, q_nbest: mp.Queue, models=None):

    args.gpu = pid
    # only support one node
    args.rank = pid

    if args.cpu:
        device = 'cpu'
        torch.set_num_threads(args.thread_per_woker)
        model, ext_lm = models
    else:
        device = pid
        torch.cuda.set_device(device)
        model, ext_lm = build_model(args, device)

    est_ilm = (args.ilm_weight != 0.)
    searcher = BeamSearcher(
        predictor=model.predictor,
        joiner=model.joiner,
        blank_id=0,
        bos_id=model.bos_id,
        beam_size=args.beam_size,
        nbest=args.beam_size,
        lm_module=ext_lm,
        alpha=args.alpha,
        beta=args.beta,
        est_ilm=est_ilm,
        ilm_weight=args.ilm_weight)

    tokenizer = tknz.load(args.tokenizer)
    nbest = {}
    with torch.no_grad(), autocast(enabled=(True if device != 'cpu' else False)):
        while True:
            batch = q_data.get(block=True)
            if batch is None:
                break
            key, x, x_lens = batch
            x = x.to(device)
            if args.streaming:
                batched_output = searcher(*(model.chunk_infer(x, x_lens))[:2])
            else:
                batched_output = searcher(*(model.encoder(x, x_lens))[:2])
            for k, (hypos, scores) in zip(key, batched_output):
                nbest[k] = {
                    nid: (scores[nid], tokenizer.decode(hypos[nid]))
                    for nid in range(len(hypos))
                }

            q_nbest.put(nbest, block=True)
            nbest = {}
            del batch

    q_nbest.put(None, block=True)


def build_model(args, device) -> Tuple[torch.nn.Module, Union[torch.nn.Module, None]]:

    if isinstance(device, int):
        device = f'cuda:{device}'
    if args.unified:
        from .train_unified import build_model as rnnt_builder
    else:
        from .train import build_model as rnnt_builder
    model = rnnt_builder(
        coreutils.readjson(args.config),
        args, dist=False)
    model = model.to(device)
    assert args.resume is not None, "Trying to decode with uninitialized parameters. Add --resume"

    model = coreutils.load_checkpoint(model, args.resume)
    model.eval()

    if args.lm_config is None:
        return model, None
    else:
        lm_configures = coreutils.readjson(args.lm_config)
        ext_lm_model = lm_builder(lm_configures, args, dist=False)
        if args.lm_check is not None:
            if os.path.isfile(args.lm_check):
                coreutils.load_checkpoint(
                    ext_lm_model.to(device), args.lm_check)
            else:
                print(f"warning: --lm-check={args.lm_check} does not exist. \n"
                      "skip loading params. this is OK if the model is NGram.")
        ext_lm_model = ext_lm_model.lm
        ext_lm_model.eval()
        return model, ext_lm_model


def _parser():
    parser = coreutils.basic_trainer_parser(
        prog='RNN-Transducer decoder.',
        training=False,
        isddp=False
    )

    parser.add_argument("--lm-config", type=str, default=None,
                        help="Config of external LM.")
    parser.add_argument("--lm-check", type=str, default=None,
                        help="Checkpoint of external LM.")
    parser.add_argument("--alpha", type=float, default=0.,
                        help="Weight of external LM.")
    parser.add_argument("--beta", type=float, default=0.,
                        help="Penalty value of external LM.")
    parser.add_argument("--ilm-weight", type=float, default=0.,
                        help="ILM weight."
                        "ilm weight != 0 would enable internal language model estimation. "
                        "This would slightly slow down the decoding.")

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default='./decode')
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer model file. See cat/shared/tokenizer.py for details.")
    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--thread-per-woker", type=int, default=1)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--silent", action='store_true', default=False)
    parser.add_argument("--unified", action='store_true', default=False)
    parser.add_argument("--streaming", action='store_true', default=False)
    return parser


if __name__ == '__main__':
    main()
