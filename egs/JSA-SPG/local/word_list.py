# Copyright 2025 Tsinghua SPMI Lab
# Author: Author: Sardar (sar_dar@foxmail.com)

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/..'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="text file.", nargs='+')
    parser.add_argument("--dir", type=str, default=None, help="lm exp dir.")
    parser.add_argument("--out", type=str, default=None, help="path of output txt list file.")
    parser.add_argument("--dump2json", action="store_true", default=False, help="whether dump to hyper-p.json file.")
    args = parser.parse_args()

    if args.dir:
        assert os.path.isdir(args.dir), f"{args.dir} is not exist!"
        import json
        from utils.pipeline._constants import F_HYPER_CONFIG, F_DATAINFO
        hyper_file = os.path.join(args.dir, F_HYPER_CONFIG)
        assert os.path.isfile(hyper_file), f"{hyper_file} is not exist!"
        assert os.path.isfile(F_DATAINFO), f"{F_DATAINFO} is not exist!"
        with open(hyper_file, "r") as ft:
            train_set = json.load(ft)['data']['train']
        with open(F_DATAINFO, "r") as fd:
            data_info = json.load(fd)

        if train_set is str:
            text = data_info[train_set]['trans']
        else:
            text = []
            for _set in train_set:
                text.append(data_info[_set]['trans'])
    else:
        text = args.text

    assert text is not None, f"A text file is required for generating a word list!"
    out = args.out if args.out else os.path.join(os.path.dirname(text[0]), "word_list.txt")
    if os.path.exists(out):
        print(f"word list exists: {out}\n... skip generating a word list!")
        sys.exit(0)

    if text is str:
        text = [text]

    word_list = set()
    for file in text:
        assert os.path.isfile(file), f"{file} is not a valid file."
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                word_seq = line.strip().split('\t', maxsplit=1)[1]
                word_list.update(word_seq.split())
    
    with open(out, "w", encoding="utf-8") as wf:
        for txt in word_list:
            if txt != ' ':
                wf.write(txt + "\n")
    
    if args.dump2json:
        assert os.path.isdir(args.dir) and os.path.isfile(os.path.join(args.dir, F_HYPER_CONFIG)), f"{args.dir} is not valid!"
        hyper_file = os.path.join(args.dir, F_HYPER_CONFIG)
        with open(hyper_file, "r") as fi:
            hyp_cfg = json.load(fi)
        hyp_cfg["tokenizer"]["option-init"]["dmap"] = out
        with open(hyper_file, "w") as fo:
            json.dump(hyp_cfg, fo, indent=4)

    