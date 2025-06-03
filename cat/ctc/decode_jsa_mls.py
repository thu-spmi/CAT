# Copyright 2025 Tsinghua University
# Apache 2.0.
# Author: Sardar (sar_dar@163.com)

""" Marginal likelihood scoring for SPG model
    NLL is decision distribution
"""

from ..shared import tokenizer as tknz
from ..shared import coreutils
from ..shared.encoder import AbsEncoder
from ..shared.data import ScpDataset, sortedScpPadCollate
from ..utils.pipeline._constants import F_HYPER_CONFIG

import os
import time
import pickle
import argparse
from tqdm import tqdm
from typing import *

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    if args.tokenizer is None or not os.path.isfile(args.tokenizer):
        raise FileNotFoundError(
            "Invalid BPE tokenizer model file: {}".format(args.tokenizer)
        )
    if args.char_tokenizer is None or not os.path.isfile(args.char_tokenizer):
        raise FileNotFoundError(
            "Invalid Character tokenizer model file: {}".format(args.char_tokenizer)
        )

    world_size = torch.cuda.device_count()
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

    mp.spawn(worker, nprocs=world_size, args=(args, q_data, q_out, None))

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
        num_workers=args.world_size,
        collate_fn=sortedScpPadCollate(),
    )

    t_beg = time.time()
    for batch in tqdm(
        testloader, desc="Marginal likelihood scoring for SPG model", total=len(testloader), leave=False
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
    device = pid
    torch.cuda.set_device(device)
    model = build_model(args).cuda(device)
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    
    bpe_tokenizer = tknz.load(args.tokenizer)
    char_tokenizer = tknz.load(args.char_tokenizer)

    assert args.n_samples > 0, "n_samples argument must be set in inference::infer::option."

    hy = {}
    # nbest file which generated from subword WFST decoding, lm_score = log(p(g))
    with open(args.nbest_file, "rb") as f_hy:
        # l_hy type: Dict[str, Dict[int, Tuple[float, str]]]
        # {'uid': {0: (-10.0, 'a b c'), 1: (-12.5, 'a b c d')}}
        l_hy = pickle.load(f_hy)
        for key, nbest in l_hy.items():
            hy[key] = list(nbest.values())
    ctc_loss = torch.nn.CTCLoss(reduction='none')

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
            
            if len(hy[key]) == 1:
                p_list = [hy[key][0][0]]
                indices = [0]
            else:
                y_bpe = [torch.tensor(bpe_tokenizer.encode(word_seq), dtype=torch.int32, device=device) for _, word_seq in hy[key]]
                ly = torch.tensor([y.shape[0] for y in y_bpe], dtype=torch.int32, device=device)
                y, ly = model.validate_zlen_and_pad(y_bpe, ly)
                y_char = [torch.tensor(char_tokenizer.encode(word_seq), dtype=torch.int32, device=device) for _, word_seq in hy[key]]
                ly_char = torch.tensor([y.shape[0] for y in y_char], dtype=torch.int32, device=device)
                y_char, ly_char = model.validate_zlen_and_pad(y_char, ly_char)
                

                log_lm_score = torch.tensor([lm_score for lm_score, _ in hy[key]], dtype=torch.float64, device=device)
                log_lm_score = torch.log(torch.pow(10, log_lm_score))
                p_list = torch.Tensor().to(device)
                logits1 = logits1.transpose(0, 1)

                # calculate NLL use n samples
                logits_g2p_enc, logits_lens_g2p_enc = model.g2p_encoder(y_char, ly_char)
                logits_g2p_enc = torch.log_softmax(logits_g2p_enc, dim=-1)
                samples, sample_lens = model._sample(logits_g2p_enc.detach().exp(), logits_lens_g2p_enc, n_samples=args.n_samples)
                N, K, T = samples.shape
                batch_sample_lens = sample_lens.view(N*K).to(device)
                batch_samples = torch.split(samples.view(N*K,T), torch.max(batch_sample_lens), dim=1)[0].to(device)
                logits_g2p_enc = logits_g2p_enc.transpose(0, 1)
                
                logits, olens = model.p2g_encoder(batch_samples, batch_sample_lens)
                logits = torch.log_softmax(logits, dim=-1).transpose(0, 1)
                p2g_loss = ctc_loss(logits, y.repeat_interleave(K, dim=0), olens.to(torch.int), ly.repeat_interleave(K, dim=0)).to(torch.float64)
                s2p_loss = ctc_loss(logits1.repeat_interleave(N*K, dim=1), batch_samples, olens1.repeat(N*K), batch_sample_lens).to(torch.float64)
                g2p_loss = ctc_loss(logits_g2p_enc.repeat_interleave(K, dim=1), batch_samples, logits_lens_g2p_enc.repeat_interleave(K, dim=0).to(torch.int), batch_sample_lens)
                
                log_p = g2p_loss - s2p_loss - p2g_loss
                acc_log_p = torch.logsumexp(log_p.view(N,K), 1) - torch.log(torch.tensor([args.n_samples], dtype=torch.float64, device=device))
                p_list = acc_log_p + log_lm_score * args.lm_weight

                _, indices = torch.sort(-p_list)
                p_list = p_list.tolist()

            q_out.put(
                (key, {0: (p_list[indices[0]], hy[key][indices[0]][1])}),
                block=True,
            )
            del batch

    q_out.put(None, block=True)

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
    parser.add_argument(
        "--lm_weight",
        type=float,
        default=0.0,
        help="The LM weight value for LM integration.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer model file. See cat/shared/tokenizer.py for details.",
    )
    parser.add_argument(
        "--char_tokenizer",
        type=str,
        default=None,
        help="Character tokenizer model file. See cat/shared/tokenizer.py for details.",
    )
    parser.add_argument("--thread-per-woker", type=int, default=1)
    parser.add_argument(
        "--built-model-by",
        type=str,
        default="cat.ctc.train_jsa",
        help="Tell where to import build_model() function. defautl: cat.ctc.train_jsa",
    )
    parser.add_argument("--n_samples", type=int, default=-1, help="number of samples for MIS sampling.")
    return parser

if __name__ == "__main__":
    main()
