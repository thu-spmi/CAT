# Copyright 2025 Tsinghua SPMI Lab
# Author: Author: Sardar (sar_dar@foxmail.com)

import argparse
import os
import sys
import re
import pickle
from typing import List, Union
import cat.shared.tokenizer as tknz
from cat.utils.pipeline.common_utils import checkExist, TextUtterancesOrdered, sfmt


def pack_text_data(
    f_inputs: List[str],
    f_labels: List[str],
    f_out: str,
    is_nbest_input: bool,
    is_DANP: bool
):
    
    import numpy as np
    from tqdm import tqdm

    if os.path.isfile(f_out):
        sys.stderr.write(
            sfmt.warn(
                f"file exist: {sfmt.udl(f_out)}, "
                "rm it if you want to update the data.\n",
                pack_text_data,
            )
        )
        return

    checkExist("f", f_inputs + f_labels)
    
    phoneme_list = [
        "a˥", "a˥˩", "a˧˥", "a˧˩˧",
        "e˥", "e˥˩", "e˧˥", "e˧˩˧",
        "i˥", "i˥˩", "i˧˥", "i˧˩˧",
        "o˥", "o˥˩", "o˧˥", "o˧˩˧",
        "u˥", "u˥˩", "u˧˥", "u˧˩˧",
        "y˥", "y˥˩", "y˧˥", "y˧˩˧",
        "ə˥", "ə˥˩", "ə˧˥", "ə˧˩˧",
        "ɛ˥", "ɛ˥˩", "ɛ˧˥", "ɛ˧˩˧",
        "ɤ˥", "ɤ˥˩", "ɤ˧˥", "ɤ˧˩˧",
        "ɹ̩˥", "ɹ̩˥˩", "ɹ̩˧˥", "ɹ̩˧˩˧",
        "ɻ̩˥", "ɻ̩˥˩", "ɻ̩˧˥", "ɻ̩˧˩˧",
        "ʊ˥", "ʊ˥˩", "ʊ˧˥", "ʊ˧˩˧"
    ]
    phoneme_list_replace = [
        "ā", "à", "á", "ǎ",
        "ē", "è", "é", "ě",
        "ī", "ì", "í", "ǐ",
        "ō", "ò", "ó", "ǒ",
        "ū", "ù", "ú", "ǔ",
        "ȳ", "ỳ", "ý", "y̌",
        "ə̄", "ə̀", "ə́", "ə̌",
        "ɛ̄", "ɛ̀", "ɛ́", "ɛ̌",
        "ɤ̄", "ɤ̀", "ɤ́", "ɤ̌",
        "ɹ̩̄", "ɹ̩̀", "ɹ̩́", "ɹ̩̌",
        "ɻ̩̄", "ɻ̩̀", "ɻ̩́", "ɻ̩̌",
        "ʊ̄", "ʊ̀", "ʊ́", "ʊ̌"
    ]
    replace_dict = dict(zip(phoneme_list, phoneme_list_replace))
    sorted_phonemes = sorted(replace_dict.keys(), key=lambda x: -len(x))
    pattern = re.compile(
        r'(\s|^)' + r'(' + '|'.join(re.escape(p) for p in sorted_phonemes) + r')' + r'(?=\s|$)'
    )

    def replace_tones(phn_seq: str) -> str:
        def replace_phoneme(match):
            prefix = match.group(1)
            phoneme = match.group(2)
            return prefix + replace_dict.get(phoneme, phoneme)
        
        return pattern.sub(replace_phoneme, phn_seq)

    # Read label files.
    twrapper_label = TextUtterancesOrdered(f_labels)

    inputs = []
    outputs = []
    cnt = 0

    if is_nbest_input:
        nbest_list = {}
        for nbest_file in f_inputs:
            checkExist("f", nbest_file)
            with open(nbest_file, "rb") as f_hy:
                # type: Dict[str, Dict[int, Tuple[float, str]]]
                nbest_list.update(pickle.load(f_hy))
        
        for uid, label in tqdm(twrapper_label, total=len(twrapper_label), leave=False):
            if label == "":
                sfmt.warn(f"skip empty utt: {uid}", pack_text_data)
                continue
            if uid not in nbest_list:
                sfmt.warn(f"skip utt not in nbest list: {uid}", pack_text_data)
                continue
            phn_seq_list = [replace_tones(nbest).replace(' ', '') for _, nbest in nbest_list[uid].values()]
            if is_DANP:
                for phn_seq in phn_seq_list:
                    inputs.append([phn_seq])
                    outputs.append(label)
                    cnt += 1
            else:
                inputs.append(phn_seq_list)
                outputs.append(label)
                cnt += 1

    else:
        twrapper_scp = TextUtterancesOrdered(f_inputs)
        assert len(twrapper_scp) == len(twrapper_label), sfmt.error(
            "f_scp and f_label should match on the # of lines, "
            f"instead {len(twrapper_scp)} != {len(twrapper_label)}",
            pack_text_data,
        )
    
        for (uid, label), (uid1, phn) in tqdm(zip(twrapper_label, twrapper_scp), total=len(twrapper_label), leave=False
        ):
            assert uid == uid1, f"UID in label and scp files mismatch: {uid} != {uid1}"
            if label == "":
                sfmt.warn(f"skip empty utt: {uid}", pack_text_data)
                continue

            phn_seq = replace_tones(phn).replace(' ', '')
            inputs.append([phn_seq])
            outputs.append(label)
            cnt += 1

    if cnt == 0:
        sys.stderr.write(sfmt.error("no qualified seq found.\n", pack_text_data))
        sys.exit(1)

    with open(f_out, "wb") as fo:
        pickle.dump(
            {
                "input": inputs,
                "label": outputs,
            },
            fo,
        )

    cntrm = len(twrapper_label) - cnt
    if cntrm > 0:
        print(f"pack_data(): remove {cntrm} unqualified sequences.")
    print(f"### pickled {cnt} seqs in {f_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("input_tokenizer_file", type=str, help="path of .tknz tokenizer file")
    # parser.add_argument("output_tokenizer_file", type=str, help="path of .tknz tokenizer file")
    parser.add_argument("out", type=str, help="path of output pkl file")
    parser.add_argument("--input_files", nargs="+", default=None, help="path of train set nbest file for Pickle data")
    parser.add_argument("--output_files", nargs="+", default=None, help="path of dev set nbest file for Pickle data")
    parser.add_argument("--nbest_input", action='store_true', help="is nbest input or not")
    parser.add_argument("--DANP", action='store_true', help="is DANP model or not. for DANP model, data is one input to one output. for others, data is n to one.")
    # parser.add_argument("--test", action='store_true', help="pkl data from test set")
    
    args = parser.parse_args()

    # checkExist("f", args.input_tokenizer_file)
    # input_tokenizer = tknz.load(args.input_tokenizer_file)
    # assert isinstance(input_tokenizer, tknz.LexiconTokenizer), "only LexiconTokenizer is supported for Qwen-P2G input tokenizer."
    # checkExist("f", args.output_tokenizer_file)
    # output_tokenizer = tknz.load(args.output_tokenizer_file)
    # assert isinstance(output_tokenizer, tknz.PretrainedTokenizer), "only PretrainedTokenizer is supported for Qwen-P2G output tokenizer."
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(sfmt.header("Stage 2 Pickle data"))
    fmt = sfmt(sfmt("Pickle data: ", sfmt.BOLD), sfmt.OKCYAN) + "{}\n"
    
    pack_text_data(
        args.input_files,
        args.output_files,
        f_out=args.out,
        is_nbest_input=args.nbest_input,
        is_DANP=args.DANP
    )
            
    