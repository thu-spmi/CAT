'''
Author: Huahuan Zheng
This script used to compute WER of setences.
'''

import jiwer
import argparse
import os
import sys
import re
import pickle
from multiprocessing import Pool
from typing import *


class Processor():
    def __init__(self) -> None:
        self._process = []

    def append(self, new_processing: Callable[[str], str]):
        self._process.append(new_processing)
        pass

    def __call__(self, seqs: Union[List[str], str]) -> Union[List[str], str]:

        if isinstance(seqs, str):
            for processing in self._process:
                seqs = processing(seqs)
            return seqs
        else:
            o_seq = seqs
            for processing in self._process:
                o_seq = [processing(s) for s in o_seq]
            return o_seq


def WER(l_gt: List[str], l_hy: List[str]) -> Tuple[int, int, int, int, int]:
    """Compute WER and SER from given list of sentences
    
    Args:
        l_gt (list(str)): list of ground truth sentences
        l_hy (list(str)): list of hypothesis sentences
    
    Returns:
        (sub, del, ins, hit, n_se)
        sub (int): sum of of substitutions
        del (int): sum of of deletions
        ins (int): sum of of insertions
        hit (int): sum of of hits
        n_se (int): number of mismatch sentences
    """
    assert len(l_gt) == len(l_hy)
    measures = jiwer.compute_measures(l_gt, l_hy)

    cnt_err_utt = sum(1 for gt, hy in zip(l_gt, l_hy) if gt != hy)

    return measures['substitutions'], measures['deletions'], measures['insertions'], measures['hits'], cnt_err_utt


def oracleWER(l_gt: List[Tuple[str, str]], l_hy: List[Tuple[str, List[str]]]) -> Tuple[int, int, int, int, int]:
    """Computer oracle WER.

    Take first col of l_gt as key

    Returns have the same meaning as returns of WER()
    """

    l_hy = {key: nbest for key, nbest in l_hy}
    _sub, _del, _ins, _hit, _se = 0, 0, 0, 0, 0
    for key, g_s in l_gt:
        candidates = l_hy[key]
        best_wer = float('inf')
        best_measure = {}

        mismatch = 1
        for can_seq in candidates:
            if can_seq == g_s:
                mismatch = 0
            part_ith_measure = jiwer.compute_measures(g_s, can_seq)
            if part_ith_measure['wer'] < best_wer:
                best_wer = part_ith_measure['wer']
                best_measure = part_ith_measure

        _sub += best_measure['substitutions']
        _del += best_measure['deletions']
        _ins += best_measure['insertions']
        _hit += best_measure['hits']
        _se += mismatch

    return _sub, _del, _ins, _hit, _se


def run_wer_wrapper(args):
    return WER(*args)


def run_oracle_wer_wrapper(args):
    return oracleWER(*args)


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    ground_truth = args.gt  # type:str
    hypothesis = args.hy  # type:str
    assert os.path.isfile(ground_truth), ground_truth
    assert os.path.isfile(hypothesis), hypothesis

    with open(ground_truth, 'r') as f_gt:
        l_gt = f_gt.readlines()

    if args.oracle:
        with open(hypothesis, 'rb') as f_hy:
            # type: Dict[str, Dict[int, Tuple[float, str]]]
            l_hy = pickle.load(f_hy)
        l_hy = [(key, list(nbest.values())) for key, nbest in l_hy.items()]
    else:
        try:
            with open(hypothesis, 'r') as f_hy:
                l_hy = f_hy.readlines()
        except UnicodeDecodeError:
            print(
                "Error:\n"
                f"seems the given hypothesis: '{hypothesis}' is not a text file.\n"
                f"... add --oracle if you want to compute oracle error rate.")
            sys.exit(1)

    num_lines = len(l_gt)
    assert num_lines == len(
        l_hy), f"# lines mismatch in ground truth and hypothesis files: {num_lines} != {len(l_hy)}"

    # Pre-processing
    processor = Processor()

    # replace '\t' to space
    processor.append(lambda s: s.replace('\t', ' '))

    # rm consecutive spaces
    pattern = re.compile(r' {2,}')
    processor.append(lambda s: pattern.sub(' ', s))

    # rm the '\n' and the last space
    processor.append(lambda s: s.strip('\n '))

    if args.cer:
        # rm space then split by char
        pattern = re.compile(r'\s+')
        processor.append(lambda s: pattern.sub('', s))
        processor.append(lambda s: ' '.join(list(s)))

    if args.force_cased:
        processor.append(lambda s: s.lower())

    if args.oracle:
        for i, hypo in enumerate(l_hy):
            key, nbest = hypo
            seqs = processor([s for _, s in nbest])
            l_hy[i] = (key, seqs)

        for i, s in enumerate(l_gt):
            key, g_s = s.split(maxsplit=1)
            l_gt[i] = (key, processor(g_s))

        l_hy = sorted(l_hy, key=lambda item: item[0])
        l_gt = sorted(l_gt, key=lambda item: item[0])
    elif args.noid:
        l_hy = processor(l_hy)
        l_gt = processor(l_gt)
    else:
        for i, s in enumerate(l_gt):
            key, g_s = s.split(maxsplit=1)
            l_gt[i] = (key, processor(g_s))

        for i, s in enumerate(l_hy):
            try:
                key, g_s = s.split(maxsplit=1)
            except ValueError:
                # sentence is empty
                key = s.strip()
                g_s = ''
            l_hy[i] = (key, processor(g_s))

        l_hy = sorted(l_hy, key=lambda item: item[0])
        l_gt = sorted(l_gt, key=lambda item: item[0])
        l_hy = [seq for _, seq in l_hy]
        l_gt = [seq for _, seq in l_gt]

    # multi-processing compute
    num_threads = max(min(num_lines//10000, int(os.cpu_count())), 1)

    interval = num_lines // num_threads
    indices = [interval * i for i in range(num_threads+1)]
    if indices[-1] != num_lines:
        indices[-1] = num_lines
    pool_args = [(l_gt[indices[i]:indices[i+1]], l_hy[indices[i]:indices[i+1]])
                 for i in range(num_threads)]

    with Pool(processes=num_threads) as pool:
        if args.oracle:
            gathered_measures = pool.map(run_oracle_wer_wrapper, pool_args)
        else:
            gathered_measures = pool.map(run_wer_wrapper, pool_args)

    # gather sub-processes results
    _sub, _del, _ins, _hits, _se = 0, 0, 0, 0, 0
    for p_sub, p_del, p_ins, p_hits, p_se in gathered_measures:
        _sub += p_sub
        _del += p_del
        _ins += p_ins
        _hits += p_hits
        _se += p_se

    _err = _sub + _del + _ins
    _sum = _hits + _sub + _del
    _wer = _err / _sum
    _ser = _se / num_lines

    # format: %SER 13.60 | %WER 4.50 [ 2367 / 52576, 308 ins, 157 del, 1902 sub ]
    prefix = 'WER' if not args.cer else 'CER'
    pretty_str = \
        f"%SER {_ser*100:.2f} | %{prefix} {_wer*100:.2f} [ {_err} / {_sum}, {_ins} ins, {_del} del, {_sub} sub ]"

    sys.stdout.write(pretty_str+'\n')
    return {
        'ser': _ser,
        'wer': _wer,
        'ins': _ins,
        'del': _del,
        'sub': _sub,
        'string': pretty_str}


def _parser():
    parser = argparse.ArgumentParser('Compute WER/CER')
    parser.add_argument("gt", type=str, help="Ground truth sequences.")
    parser.add_argument("hy", type=str, help="Hypothesis of sequences.")
    parser.add_argument("--noid", action="store_true", default=False,
                        help="Process the text as raw without utterance id. When --oracle, this will be ignored.")
    parser.add_argument("--cer", action="store_true", default=False,
                        help="Compute CER. Default: False")
    parser.add_argument("--force-cased", action="store_true",
                        help="Force text to be the same cased.")
    parser.add_argument("--oracle", action="store_true", default=False,
                        help="Compute Oracle WER/CER. This requires the `hy` to be N-best list instead of text. Default: False")
    return parser


if __name__ == "__main__":
    main()
