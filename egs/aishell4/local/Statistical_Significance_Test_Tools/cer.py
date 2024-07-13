# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Xiangzhu Kong (kongxiangzhu99@gmail.com)

"""Compute WER/CER of sentences.
"""

import jiwer
import argparse
import os
import sys
import re
from multiprocessing import Pool
from typing import *
import json


class Processor:
    """
    A class used to process strings or lists of strings through a series of user-defined processing functions.

    Methods:
        append(new_processing: Callable[[str], str]):
            Adds a new processing function to the processing pipeline.

        __call__(seqs: Union[List[str], str]) -> Union[List[str], str]:
            Applies the processing pipeline to the input string or list of strings.
    """
    def __init__(self) -> None:
        self._process = []

    def append(self, new_processing: Callable[[str], str]):
        self._process.append(new_processing)

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


def CER(gt: str, hy: str) -> float:
    """Compute CER from given ground truth and hypothesis sentences

    Args:
        gt (str): ground truth sentence
        hy (str): hypothesis sentence

    Returns:
        float: Character Error Rate
    """
    measures = jiwer.compute_measures(gt, hy)
    return measures["wer"]


def run_cer_wrapper(args):
    return CER(*args)


def main(args: argparse.Namespace = None):
    if args is None:
        parser = _parser()
        args = parser.parse_args()

    ground_truth = args.gt  # type:str
    hypothesis = args.hy  # type:str
    assert os.path.isfile(ground_truth), ground_truth
    assert os.path.isfile(hypothesis), hypothesis

    with open(ground_truth, "r",encoding='utf-8') as f_gt:
        l_gt = f_gt.readlines()

    try:
        with open(hypothesis, "r",encoding='utf-8') as f_hy:
            l_hy = f_hy.readlines()
    except UnicodeDecodeError:
        print(
            "Error:\n"
            f"seems the given hypothesis: '{hypothesis}' is not a text file.\n"
        )
        sys.exit(1)

    num_lines = len(l_gt)
    assert num_lines == len(
        l_hy
    ), f"# lines mismatch in ground truth and hypothesis files: {num_lines} != {len(l_hy)}"

    # Pre-processing
    processor = Processor()

    # replace '\t' to space
    processor.append(lambda s: s.replace("\t", " "))

    # rm consecutive spaces
    pattern = re.compile(r" {2,}")
    processor.append(lambda s: pattern.sub(" ", s))

    # rm the '\n' and the last space
    processor.append(lambda s: s.strip("\n "))

    if args.cer:
        # rm space then split by char
        pattern = re.compile(r"\s+")
        processor.append(lambda s: pattern.sub("", s))
        processor.append(lambda s: " ".join(list(s)))

    if args.force_cased:
        processor.append(lambda s: s.lower())

    # 处理 l_gt，将每行的第一个词与后面的文本分开，得到键（key）和正确文本
    for i, s in enumerate(l_gt):
        key, g_s = s.split(maxsplit=1)
        l_gt[i] = (key, processor(g_s))

    # 处理 l_hy，将每行的第一个词与后面的文本分开，得到键（key）和处理后的文本
    for i, s in enumerate(l_hy):
        try:
            key, g_s = s.split(maxsplit=1)
        except ValueError:
            # sentence is empty
            key = s.strip()
            g_s = ""
        l_hy[i] = (key, processor(g_s))

    # 排序处理后的 l_gt 和 l_hy
    l_hy = sorted(l_hy, key=lambda item: item[0])
    l_gt = sorted(l_gt, key=lambda item: item[0])
    #keys_gt = [key for key, _ in l_gt]
    l_hy = [seq for _, seq in l_hy]
    l_gt = [seq for _, seq in l_gt]
    # Process each sentence and calculate CER
    cer_results = []
    for i in range(num_lines):
        # Split each line into key and sentence
        gt_sentence = l_gt[i]
        hy_sentence = l_hy[i]

        cer = CER(gt_sentence, hy_sentence)
        #cer = 1 if cer > 0 else 0
        #cer_results.append(keys_gt[i] + ' ' + str(cer))
        cer_results.append(cer)
        


    # Save CER results to JSON file
    output_file_path = args.output_path
    with open(output_file_path, "w") as json_file:
        json.dump(cer_results, json_file, indent=2)

    print(f"CER results saved to {output_file_path}")

    return cer_results


def _parser():
    parser = argparse.ArgumentParser("Compute WER/CER")
    parser.add_argument("gt", type=str, help="Ground truth sequences.")
    parser.add_argument("hy", type=str, help="Hypothesis of sequences.")
    parser.add_argument(
        "--cer", action="store_true", default=False, help="Compute CER. Default: False"
    )
    parser.add_argument(
        "--force-cased", action="store_true", help="Force text to be the same cased."
    )
    parser.add_argument(
        "--output-path", type=str, default="cer_results.json", help="Path to save the CER results JSON file."
    )
    return parser


if __name__ == "__main__":
    main()
