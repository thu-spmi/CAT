"""CTC decode module

NOTE (Huahuan):
    Currently, bs=1 is hard-coded.
Derived from
https://github.com/parlance/ctcdecode
"""

from ..shared import tokenizer as tknz
from ..shared import coreutils
from ..shared.encoder import AbsEncoder
from ..shared.data import (
    ScpDataset,
    sortedScpPadCollate
)


import os
import time
import pickle
import argparse
from tqdm import tqdm
from typing import *
from ctcdecode import CTCBeamDecoder

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    if args.tokenizer is None or not os.path.isfile(args.tokenizer):
        raise FileNotFoundError(
            "Invalid tokenizer model file: {}".format(args.tokenizer))

    if args.gpu:
        world_size = torch.cuda.device_count()
        if args.nj != -1 and args.nj < world_size:
            world_size = args.nj
    else:
        if args.nj == -1:
            world_size = max(os.cpu_count()//2, 1)
        else:
            world_size = args.nj
    assert world_size > 0
    args.world_size = world_size

    try:
        mp.set_start_method('spawn')
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
        testset, batch_size=1, shuffle=False,
        num_workers=(args.world_size if args.gpu else args.world_size//8),
        collate_fn=sortedScpPadCollate())

    t_beg = time.time()
    for batch in tqdm(testloader, desc="CTC decode", total=len(testloader), leave=False):
        for k in batch:
            if isinstance(k, torch.Tensor):
                k.share_memory_()
        q.put(batch, block=True)

    for _ in range(args.world_size+1):
        q.put(None, block=True)
    t_dur = time.time() - t_beg

    print("Time = {:.2f} s | RTF = {:.2f} ".format(
        t_dur, t_dur*args.world_size / n_frames * 100))


def datawriter(args, q: mp.Queue):
    cnt_done = 0
    nbest = {}
    transcript = []

    while True:
        # type: Tuple[str, Dict[int, Tuple[float, str]]]
        nbestlist = q.get(block=True)
        if nbestlist is None:
            cnt_done += 1
            if cnt_done == args.world_size:
                break
            continue
        key, content = nbestlist
        nbest[key] = content
        transcript.append(f"{key}\t{content[0][1]}\n")
        del nbestlist

    with open(args.output_prefix, 'w') as fo:
        for l in transcript:
            fo.write(l)

    with open(args.output_prefix+'.nbest', 'wb') as fo:
        pickle.dump(nbest, fo)


def worker(pid: int, args: argparse.Namespace, q_data: mp.Queue, q_out: mp.Queue, model: AbsEncoder):
    torch.set_num_threads(args.thread_per_woker)
    if args.gpu:
        device = pid
        torch.cuda.set_device(device)
        model = build_model(args).cuda(device)
    else:
        assert model is not None
        device = 'cpu'

    tokenizer = tknz.load(args.tokenizer)
    if args.lm_path is None:
        # w/o LM, labels won't be used in decoding.
        labels = [''] * tokenizer.vocab_size
        searcher = CTCBeamDecoder(
            labels, beam_width=args.beam_size, log_probs_input=True, num_processes=args.thread_per_woker)
    else:
        assert os.path.isfile(
            args.lm_path), f"--lm-path={args.lm_path} is not a valid file."

        # NOTE: ctc decoding with an ext. LM assumes <s> -> 0 and <unk> -> 1
        labels = ['<s>', '<unk>'] + [str(i)
                                     for i in range(2, tokenizer.vocab_size)]
        searcher = CTCBeamDecoder(
            labels, model_path=args.lm_path, alpha=args.alpha, beta=args.beta,
            beam_width=args.beam_size, num_processes=args.thread_per_woker,
            log_probs_input=True, is_token_based=True)

    # {'uid': {0: (-10.0, 'a b c'), 1: (-12.5, 'a b c d')}}
    nbest = {}  # type: Dict[str, Dict[int, Tuple[float, str]]]
    with torch.no_grad():
        while True:
            batch = q_data.get(block=True)
            if batch is None:
                break
            key, x, x_len = batch
            x = x.to(device)
            key = key[0]
            if args.streaming:
                logits, olens = model.chunk_infer(x, x_len)
            else:
                logits, olens = model.am(x, x_len)

            # NOTE: log_softmax makes no difference in ctc beam search
            #       however, if you would like to do further work with the AM score,
            #       you may need to do the normalization.
            if args.do_normalize:
                logits = torch.log_softmax(logits, dim=-1)

            beam_results, beam_scores, _, out_lens = searcher.decode(
                logits.cpu(), olens)
            # make it in descending order
            # -log(p) -> log(p)
            beam_scores = -beam_scores

            q_out.put(
                (key, {
                    bid: (score.item(), tokenizer.decode(
                        hypo[:_len].tolist()))
                    for bid, (score, hypo, _len) in enumerate(zip(beam_scores[0], beam_results[0], out_lens[0]))
                }), block=True)

            del batch
    q_out.put(None, block=True)


def build_model(args: argparse.Namespace):
    assert args.resume is not None, "Trying to decode with uninitialized parameters. Add --resume"
    import importlib
    interface = importlib.import_module(args.built_model_by)
    model = interface.build_model(coreutils.readjson(args.config), dist=False)
    model.clean_unpickable_objs()
    checkpoint = torch.load(args.resume, map_location='cpu')
    model = coreutils.load_checkpoint(model, checkpoint)
    model.eval()
    return model


def _parser():
    parser = coreutils.basic_trainer_parser(
        prog="CTC decoder.",
        training=False,
        isddp=False
    )

    parser.add_argument("--input_scp", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default='./decode')
    parser.add_argument("--lm-path", type=str, help="Path to KenLM model.")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="The 'alpha' value for LM integration, a.k.a. the LM weight")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="The 'beta' value for LM integration, a.k.a. the penalty of tokens.")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--do-normalize", action="store_true",
                        default=False, help="Do the log-softmax normalization before beam search.")
    parser.add_argument("--tokenizer", type=str,
                        help="Tokenizer model file. See cat/shared/tokenizer.py for details.")
    parser.add_argument("--gpu", action="store_true", default=False,
                        help="Use GPU to do inference. Default: False.")
    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("--thread-per-woker", type=int, default=1)
    parser.add_argument("--built-model-by", type=str, default="cat.ctc.train",
                        help="Tell where to import build_model() function. defautl: cat.ctc.train")
    parser.add_argument("--streaming", action='store_true', default=False)
    return parser


if __name__ == '__main__':
    main()
