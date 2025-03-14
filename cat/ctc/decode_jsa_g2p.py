# Copyright 2023 Tsinghua University
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
from ..shared.data import sortedScpPadCollate, P2GTestDataset


import os
import pickle
import kaldiio
import argparse
from tqdm import tqdm
from typing import *
from ctcdecode import CTCBeamDecoder as CTCDecoder
import ctc_align

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    if args.phone_tokenizer is None or not os.path.isfile(args.phone_tokenizer):
        raise FileNotFoundError(
            "Invalid phoneme tokenizer model file: {}".format(args.phone_tokenizer)
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
    testset = P2GTestDataset(args.input_scp)
    testloader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=(args.world_size if args.gpu else args.world_size // 8),
        collate_fn=sortedScpPadCollate(),
    )

    for batch in tqdm(
        testloader, desc="CTC decode", total=len(testloader), leave=False
    ):
        for k in batch:
            if isinstance(k, torch.Tensor):
                k.share_memory_()
        q.put(batch, block=True)

    for _ in range(args.world_size + 1):
        q.put(None, block=True)

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

    if args.save_nbest:
        with open(args.output_prefix + ".nbest", "wb") as fo:
            pickle.dump(nbest, fo)


def worker(
    pid: int,
    args: argparse.Namespace,
    q_data: mp.Queue,
    q_out: mp.Queue,
    model: AbsEncoder,
):
    torch.set_num_threads(args.thread_per_woker)
    if args.gpu:
        device = pid
        torch.cuda.set_device(device)
        model = build_model(args).cuda(device)
    else:
        assert model is not None
        device = "cpu"

    tokenizer = tknz.load(args.phone_tokenizer)
    labels = [""] * tokenizer.vocab_size
    searcher = CTCDecoder(
        labels,
        beam_width=args.beam_size,
        log_probs_input=True,
        num_processes=args.thread_per_woker,
    )

    results = {}
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    # {'uid': {0: (-10.0, 'a b c'), 1: (-12.5, 'a b c d')}}
    with torch.no_grad():
        while True:
            batch = q_data.get(block=True)
            if batch is None:
                break
            key, x, x_len = batch
            x = x.to(device)
            x_len = x_len.to(device)
            key = key[0]

            logits, olens = model.g2p_encoder(x, x_len)

            # NOTE: log_softmax makes no difference in ctc beam search
            #       however, if you would like to do further work with the AM score,
            #       you may need to do the normalization.
            if args.do_normalize:
                logits = torch.log_softmax(logits, dim=-1)

            results[key] = logits[0].cpu().numpy()
            if args.do_sample:
                if args.sample_each_utt:
                    sample_results = torch.Tensor().to(device).to(torch.int64)
                    sample_lens = torch.Tensor().to(device).to(torch.int32)
                    sample_list = set()
                    while len(sample_list) < args.n_samples:
                        beam_results, out_lens = _sample(logits.exp(), olens, 1)
                        sample = " ".join(map(str,beam_results[0][0][:out_lens[0][0]].tolist()))
                        if sample not in sample_list:
                            sample_list.add(sample)
                            sample_results = torch.cat((sample_results, beam_results), dim=1)
                            sample_lens = torch.cat((sample_lens, out_lens), dim=1)
                    N, K, T = sample_results.shape
                    sample_scores = torch.zeros([N,K], dtype=torch.float16, device=device)
                    beam_results = sample_results
                    out_lens = sample_lens
                    beam_scores = sample_scores
                else:
                    sample_results, sample_lens = _sample(logits.exp(), olens, args.n_samples)
                    N, K, T = sample_results.shape
                    batch_phn_lens = sample_lens.view(N*K).to(device)
                    phn_mask = torch.arange(torch.max(batch_phn_lens), device=device)[None, :] < batch_phn_lens[:, None]
                    batch_phn_results = torch.split(sample_results.view(N*K,T), torch.max(batch_phn_lens), dim=1)[0].to(device) * phn_mask
                    beam_results = torch.unique(batch_phn_results.view(N,K,-1), dim=1, sorted=False)
                    out_lens = torch.count_nonzero(beam_results != 0, dim=-1)
                    beam_scores = torch.zeros([N,K], dtype=torch.float16, device=device)
            else:
                beam_results, beam_scores, _, out_lens = searcher.decode(
                    logits.cpu(), olens
                )
            # make it in descending order
            # -log(p) -> log(p)
            beam_scores = -beam_scores
            # NOTE: for debug
            # print(f"{key}\t{beam_results[0][0][:out_lens[0][0]].tolist()}")
            # if len(beam_results[0][0][:out_lens[0][0]]) == 0:
            #     sys.stderr.write(f"{key}\t got empty beam results! please check the audio file!")
            #     sys.exit(1)

            q_out.put(
                (
                    key,
                    {
                        bid: (score.item(), 
                              '' if len(hypo[:_len].tolist())==0 else tokenizer.decode(hypo[:_len].tolist())
                              )
                        for bid, (score, hypo, _len) in enumerate(
                            zip(beam_scores[0], beam_results[0], out_lens[0])
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

def _sample(probs, lx, n_samples):
        N, T, V = probs.shape
        K = n_samples
        # (NT, K)
        samples = torch.multinomial(probs.view(-1, V), K, replacement=True).view(
            N, T, K
        )
        # (N, T, K) -> (N, K, T) -> (N*K, T)
        ys, ly = ctc_align.align_(
            samples.transpose(1, 2).contiguous().view(-1, T),
            # (N, ) -> (N, 1) -> (N, K) -> (N*K, )
            lx.unsqueeze(1).repeat(1, K).contiguous().view(-1),
        )
        return ys.view(N, K, T), ly.view(N, K)

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
        "--do-normalize",
        action="store_true",
        default=False,
        help="Do the log-softmax normalization before beam search.",
    )
    parser.add_argument(
        "--phone_tokenizer",
        type=str,
        help="Phoneme tokenizer model file. See cat/shared/tokenizer.py for details.",
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
    parser.add_argument("--store_ark", type=bool, default=False, help="whether store logits as ark file.")
    parser.add_argument("--do_sample", type=bool, default=False, help="do sampling or beamsearch decoding.")
    parser.add_argument("--save_nbest", type=bool, default=False, help="whether store n-best list.")

    return parser


if __name__ == "__main__":
    main()
