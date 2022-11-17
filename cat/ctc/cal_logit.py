"""
Copyright 2022 Tsinghua University
Apache 2.0.
Author: Hongyu Xiang, Keyu An, Huahuan Zheng
"""


from .decode import build_model
from ..shared import coreutils
from ..shared.encoder import AbsEncoder
from ..shared.data import (
    ScpDataset,
    sortedScpPadCollate
)


import os
import time
import kaldiio
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.nj == -1:
        world_size = os.cpu_count()
    else:
        world_size = args.nj
    assert world_size > 0
    args.world_size = world_size

    try:
        mp.set_start_method('spawn')
    except RuntimeError as re:
        print(re)

    q = mp.Queue(maxsize=1)
    producer = mp.Process(target=dataserver, args=(args, q))
    producer.start()

    model = build_model(args)
    model.share_memory()
    mp.spawn(worker, nprocs=world_size, args=(args, q, model))
    producer.join()
    del q


def dataserver(args, q: mp.Queue):
    testset = ScpDataset(args.input_scp)
    n_frames = sum(testset.get_seq_len())
    testloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=args.world_size//8,
        collate_fn=sortedScpPadCollate())

    t_beg = time.time()
    for batch in tqdm(testloader, desc="Cal logit", total=len(testloader), leave=False):
        for k in batch:
            if isinstance(k, torch.Tensor):
                k.share_memory_()
        q.put(batch, block=True)

    for _ in range(args.world_size+1):
        q.put(None, block=True)
    t_dur = time.time() - t_beg

    print("Time = {:.2f} s | RTF = {:.2f} ".format(
        t_dur, t_dur*args.world_size / n_frames * 100))


def worker(pid: int, args: argparse.Namespace, q: mp.Queue, model: AbsEncoder):

    torch.set_num_threads(1)

    results = {}
    with torch.no_grad():
        while True:
            batch = q.get(block=True)
            if batch is None:
                break
            key, x, x_lens = batch
            assert len(key) == 1, "Batch size > 1 is not currently support."
            if args.streaming:
                logits = model.chunk_infer(x, x_lens)[0].data.numpy()
            else:
                logits = model.am(x, x_lens)[0].data.numpy()
            logits[logits == -np.inf] = -1e16
            results[key[0]] = logits[0]
            del batch

    kaldiio.save_ark(os.path.join(
        args.output_dir, f"decode.{pid+1}.ark"), results)


def _parser():
    parser = coreutils.basic_trainer_parser(
        prog="CTC logit generator.",
        training=False,
        isddp=False
    )

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output-dir", type=str, help="Ouput directory.")
    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--built-model-by", type=str, default="cat.ctc.train",
                        help="Tell where to import build_model() function. defautl: cat.ctc.train")
    parser.add_argument("--streaming", action='store_true', default=False)
    return parser


if __name__ == '__main__':
    main()
