# Copyright 2025 Tsinghua University
# Apache 2.0.
# Author: Sardar (sar_dar@163.com)

"""CTC decode module

NOTE (Huahuan): currently, bs=1 is hard-coded.

Reference:
https://github.com/parlance/ctcdecode
"""

from ..shared import tokenizer as tknz
from ..shared import coreutils
from ..shared.encoder import AbsEncoder
from ..shared.data import ScpDataset, sortedScpPadCollate


import os
import time
import kaldiio
import argparse
from tqdm import tqdm
from typing import *
from ctcdecode import CTCBeamDecoder as CTCDecoder

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    if args.tokenizer is None or not os.path.isfile(args.tokenizer):
        raise FileNotFoundError(
            "Invalid tokenizer model file: {}".format(args.tokenizer)
        )

    if args.gpu:
        world_size = torch.cuda.device_count()
        if args.nj != -1 and args.nj < world_size:
            world_size = args.nj
    else:
        if args.nj == -1:
            world_size = max(os.cpu_count() // 2, 1)
        else:
            world_size = args.nj
    assert world_size > 0
    args.world_size = world_size

    try:
        mp.set_start_method("spawn")
    except RuntimeError as re:
        print(re)

    q_data = mp.Queue(maxsize=1)
    producer = mp.Process(target=dataserver, args=(args, q_data))
    producer.start()

    q_out = mp.Queue(maxsize=1)
    consumer = mp.Process(target=datawriter, args=(args, q_out))
    consumer.start()

    if args.gpu:
        model = None
    else:
        model = build_model(args)
        model.share_memory()
    mp.spawn(worker, nprocs=world_size, args=(args, q_data, q_out, model))

    producer.join()
    consumer.join()
    del q_data
    del q_out


def dataserver(args, q: mp.Queue):

    testset = ScpDataset(args.input_scp)
    n_frames = sum(testset.get_seq_len())
    testloader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=(args.world_size if args.gpu else args.world_size // 8),
        collate_fn=sortedScpPadCollate(),
    )

    t_beg = time.time()
    for batch in tqdm(
        testloader, desc="CTC decode", total=len(testloader), leave=False
    ):
        for k in batch:
            if isinstance(k, torch.Tensor):
                k.share_memory_()
        q.put(batch, block=True)

    for _ in range(args.world_size + 1):
        q.put(None, block=True)
    t_dur = time.time() - t_beg

    print(
        "Time = {:.2f} s | RTF = {:.2f} ".format(
            t_dur, t_dur * args.world_size / n_frames * 100
        )
    )


def datawriter(args, q: mp.Queue):
    cnt_done = 0
    nbest = {}
    transcript = []

    while True:
        nbestlist = q.get(block=True)  # type: Tuple[str, Dict[int, Tuple[float, str]]]
        if nbestlist is None:
            cnt_done += 1
            if cnt_done == args.world_size:
                break
            continue
        key, content = nbestlist
        nbest[key] = content
        transcript.append(f"{key}\t{content[0][1]}\n")
        del nbestlist

    with open(args.output_prefix, "w") as fo:
        for l in transcript:
            fo.write(l)

def worker(
    pid: int,
    args: argparse.Namespace,
    q_data: mp.Queue,
    q_out: mp.Queue,
    model: AbsEncoder,
):
    torch.set_num_threads(args.thread_per_woker)
    assert model is not None
    device = "cpu"

    tokenizer = tknz.load(args.tokenizer)

    # w/o LM, labels won't be used in decoding.
    labels = [""] * tokenizer.vocab_size
    searcher = CTCDecoder(
        labels,
        beam_width=args.beam_size,
        log_probs_input=True,
        num_processes=args.thread_per_woker,
    )
    
    labels = [""] * model.phn_searcher._num_labels
    phn_searcher = CTCDecoder(
        labels,
        beam_width=16,
        log_probs_input=True,
        num_processes=args.thread_per_woker,
    )
    results = {}
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    
    with torch.no_grad():
        while True:
            batch = q_data.get(block=True)
            if batch is None:
                break
            key, x, x_len = batch
            x = x.to(device)
            key = key[0]

            logits1, olens1 = model.s2p_encoder(x, x_len)
            logits1 = torch.log_softmax(logits1, dim=-1)
            olens1 = olens1.to(torch.int32)

            b_results, b_scores, _, blens = phn_searcher.decode(logits1, olens1)

            samples = b_results.transpose(0,1)[0]
            sample_lens = blens.transpose(0,1)[0]
            batch_size = x.shape[0]

            # validate samples because S2P decoder may return a empty result
            z = [samples[i][:sample_lens[i]] for i in range(batch_size)]
            current_samples, current_lens = model.validate_zlen_and_pad(z, sample_lens)
            logits, olens = model.p2g_encoder(current_samples.to(device), current_lens.to(device))
            logits = torch.log_softmax(logits, dim=-1)
            results[key] = logits[0].cpu().numpy()
            
            beam_results, beam_scores, _, out_lens = searcher.decode(
                logits.cpu(), olens
            )
            
            beam_scores = -beam_scores

            q_out.put(
                (
                    key,
                    {
                        0: (beam_scores[0][0].item(), 
                              '' if len(beam_results[0][0][:out_lens[0][0]].tolist())==0 else tokenizer.decode(beam_results[0][0][:out_lens[0][0]].tolist())
                              )
                    },
                ),
                block=True,
            )

            del batch
    q_out.put(None, block=True)
    if args.store_ark:
        output_dir = os.path.join(os.path.dirname(args.output_prefix),"ark")
        os.makedirs(output_dir, exist_ok=True)
        kaldiio.save_ark(os.path.join(output_dir, f"decode.{pid+1}.ark"), results)

@torch.no_grad()
def build_model(args: argparse.Namespace):
    assert (
        args.resume is not None
    ), "Trying to decode with uninitialized parameters. Add --resume"
    import importlib

    interface = importlib.import_module(args.built_model_by)
    model = interface.build_model(coreutils.readjson(args.config), dist=False)
    model.clean_unpickable_objs()
    checkpoint = torch.load(args.resume, map_location="cpu")
    model = coreutils.load_checkpoint(model, checkpoint)
    model.eval()
    return model


def _parser():
    parser = coreutils.basic_trainer_parser(
        prog="CTC decoder.", training=False, isddp=False
    )

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="./decode")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer model file. See cat/shared/tokenizer.py for details.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU to do inference. Default: False.",
    )
    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--thread-per-woker", type=int, default=1)
    parser.add_argument(
        "--built-model-by",
        type=str,
        default="cat.ctc.train_jsa",
        help="Tell where to import build_model() function. defautl: cat.ctc.train",
    )
    parser.add_argument("--store_ark", type=bool, default=True, help="whether store logits as ark file.")
    return parser


if __name__ == "__main__":
    main()
