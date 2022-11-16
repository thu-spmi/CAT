""" Compute ppl on specified test sets with LM.

"""

from . import lm_builder
from cat.shared import coreutils
from cat.shared.data import (
    CorpusDataset,
    sortedPadCollateLM
)

import os
import sys
import math
import time
import uuid
import shutil
import argparse
from typing import *

import gather

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    for _path in args.evaluate:
        if not os.path.isfile(_path):
            raise FileNotFoundError(f"{_path} does not exist!")

    if args.tokenizer is not None:
        assert os.path.isfile(
            args.tokenizer), f"no such tokenizer file: '{args.tokenizer}'"
        assert os.access('/tmp', os.W_OK), f"/tmp is non-writable."
        cachedir = os.path.join('/tmp', str(uuid.uuid4()))
        os.makedirs(cachedir)
    else:
        cachedir = None

    isngram = isNGram(args)
    if (not isngram) and torch.cuda.is_available():
        usegpu = True
    else:
        usegpu = False

    if args.nj == -1:
        if usegpu:
            world_size = torch.cuda.device_count()
        else:
            world_size = min(1, os.cpu_count() // 2)
    else:
        world_size = args.nj
    assert world_size > 0
    try:
        mp.set_start_method('spawn')
    except RuntimeError as re:
        if str(re) != "context has already been set":
            raise RuntimeError(str(re))

    q = mp.Queue(maxsize=1)
    consumer = mp.Process(target=consume_worker, args=(world_size, q, args))
    consumer.start()

    args.usegpu = usegpu
    processed_files = []
    for testset in args.evaluate:
        if args.tokenizer is not None:
            binfile = os.path.join(cachedir, f"{str(uuid.uuid4())}.pkl.tmp")
            text2corpusbin(testset, binfile, args.tokenizer)
        else:
            binfile = testset
        processed_files.append(binfile)

    if usegpu:
        mp.spawn(evaluate_nnlm, nprocs=world_size,
                 args=(world_size, q, args, processed_files))
    else:
        model = build_model(args, 'cpu')
        model.share_memory()
        if isngram:
            mp.spawn(evaluate_ngram, nprocs=world_size,
                     args=(world_size, q, args, processed_files, model))
        else:
            mp.spawn(evaluate_nnlm, nprocs=world_size,
                     args=(world_size, q, args, processed_files, model))

    if cachedir is not None:
        shutil.rmtree(cachedir)

    consumer.join()
    del q


@torch.no_grad()
def evaluate_ngram(pid: int, wsize: int, q: mp.Queue, args: argparse.Namespace, testsets: List[str], model):
    """Evaluate datasets and return sum of logprobs and tokens."""

    torch.set_num_threads(1)
    output = []     # type: List[Tuple[float, int]]
    prob_ilm = args.probing_ilm
    for f_data in testsets:
        testdata = CorpusDataset(f_data)
        log_probs = 0.
        n_tokens = 0
        for i in range(pid * (len(testdata) // wsize), (pid+1) * (len(testdata) // wsize)):
            in_tokens, targets = testdata[i]
            if prob_ilm:
                in_tokens = in_tokens[:-1]
                targets = targets[:-1]
            scores = model.score(in_tokens.unsqueeze(0), targets.unsqueeze(0))
            log_probs += scores
            n_tokens += in_tokens.size(0)
        output.append((log_probs.item(), n_tokens))
    q.put(output, block=True)


@torch.no_grad()
def evaluate_nnlm(pid: int, wsize: int, q: mp.Queue, args: argparse.Namespace, testsets: List[str], model=None):
    """Evaluate datasets and return sum of logprobs and tokens."""

    if args.usegpu:
        device = pid
        torch.cuda.set_device(device)
        model = build_model(args, device, verbose=(pid == 0))
    else:
        torch.set_num_threads(1)
        device = 'cpu'
        assert model is not None
    assert next(iter(model.parameters())).device == torch.device(device)

    prob_ilm = args.probing_ilm
    output = []     # type: List[Tuple[float, int]]
    for f_data in testsets:
        testdata = CorpusDataset(f_data)
        # slice the dataset to avoid duplicated
        testdata.offsets = testdata.offsets[
            pid * (len(testdata)//wsize):(pid+1)*(len(testdata)//wsize)]
        testloader = DataLoader(
            testdata, batch_size=32,
            num_workers=1,
            collate_fn=sortedPadCollateLM(flatten_target=False)
        )
        tot_log_probs = 0.
        n_tokens = 0
        for minibatch in testloader:
            in_tokens, in_lens, targets, _ = minibatch
            if prob_ilm:
                in_lens -= 1
            in_tokens = in_tokens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            tot_log_probs += model.score(in_tokens,
                                         targets, in_lens).sum(dim=0)
            n_tokens += in_lens.sum(dim=0)
        output.append((tot_log_probs.item(), n_tokens))

    q.put(output, block=True)
    time.sleep(2.)


def consume_worker(wsize: int, q: mp.Queue, args):
    output = []     # type: List[List[Tuple[float, int]]]
    for i in range(wsize):
        data = q.get(block=True)
        output.append(data)

    sys.stdout.write("ppl: ")
    for i_f, f_data in enumerate(args.evaluate):
        ppl = math.exp(
            -sum(
                output[i_worker][i_f][0]
                for i_worker in range(wsize)
            )/sum(
                output[i_worker][i_f][1]
                for i_worker in range(wsize)
            )
        )
        sys.stdout.write("  {:.2f}  |".format(ppl))
    sys.stdout.write('\n')


def text2corpusbin(f_text: str, f_bin: str, tokenizer):
    from ..utils.pipeline.common_utils import parse_args_from_var
    from ..utils.data import pack_corpus as t2b
    t2b.main(parse_args_from_var(
        t2b._parser(),
        {
            'tokenizer': tokenizer,
            'quiet': True
        },
        [f_text, f_bin]
    ))

    return


def isNGram(args):
    configures = coreutils.readjson(args.config)
    return configures['decoder']['type'] == 'NGram'


def build_model(args: argparse.Namespace, device, verbose: bool = True):
    configures = coreutils.readjson(args.config)
    isngram = (configures['decoder']['type'] == 'NGram')
    if not isngram:
        model = lm_builder(configures, dist=False, wrapper=True)
        if args.resume is None:
            if verbose:
                sys.stderr.write(
                    f"You're trying to compute ppl with un-initialized model.\n"
                    f"... ensure you know what you're doing.\n")
        else:
            coreutils.load_checkpoint(model, args.resume)
        # squeeze the wrapper
        model = model.lm
    else:
        model = lm_builder(configures, dist=False, wrapper=False)

    model.eval()
    model = model.to(device)
    return model


def _parser():
    parser = argparse.ArgumentParser(prog="Compute perplexity to evaluate LM.")
    parser.add_argument("config", type=str,
                        help="Path to the configuration file, usually 'path/to/config.json'")
    parser.add_argument("--nj", type=int, default=-1)
    parser.add_argument("-e", "--evaluate", type=str, nargs='+',
                        help="Evaluate test sets. w/o --tokenizer, -e inputs are assumed to be CorpusDataset format binary data.")
    parser.add_argument("--tokenizer", type=str,
                        help="Use tokenizer to encode the evaluation sets. If passed, would take -e inputs as text files.")
    parser.add_argument("--resume", type=str,
                        help="Path to the checkpoint of NNLM, not required for N-gram LM.")
    parser.add_argument("--probing-ilm", action="store_true",
                        help="Probing the LM as ILM, </s> would be excluded due to limitation of E2E model.")
    return parser


if __name__ == "__main__":
    main()
