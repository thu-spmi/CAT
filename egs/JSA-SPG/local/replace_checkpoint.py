# Copyright 2025 Tsinghua SPMI Lab
# Author: Sardar (sar_dar@foxmail.com)

import torch
import argparse
from typing import OrderedDict

def translate_checkpoint(state_dict: OrderedDict, old_string: str, new_string: str) -> OrderedDict:
    """Translate checkpoint of previous version of RNN-T so that it could be loaded with the new one."""
    old_string = old_string + '.'
    new_string = new_string + '.'
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if old_string in k:
            k = k.replace(old_string, new_string, 1)
            new_state_dict[k] = v
    return new_state_dict

def replace_checkpoint(origin_checkpoint: OrderedDict, new_checkpoint_path: str, model_name: str):
    new_checkpoint = torch.load(new_checkpoint_path, "cpu")["model"]  # type: OrderedDict
    new_checkpoint = translate_checkpoint(new_checkpoint, "encoder", model_name + "_encoder")
    origin_checkpoint["model"].update(new_checkpoint)
    return origin_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src_checkpoint",
        type=str,
        help="Path to the checkpoint.",
    )
    parser.add_argument(
        "--s2p", type=str, default=None, help="replace weights from checkpoint"
    )
    parser.add_argument(
        "--p2g", type=str, default=None, help="replace weights from checkpoint"
    )
    parser.add_argument(
        "--g2p", type=str, default=None, help="replace weights from checkpoint"
    )
    parser.add_argument(
        "--out", type=str, default=None, help="output path of checkpoint file"
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.src_checkpoint, "cpu")

    changed = False

    if args.s2p:
        checkpoint = replace_checkpoint(checkpoint, args.s2p, "s2p")
        changed = True
    
    if args.p2g:
        checkpoint = replace_checkpoint(checkpoint, args.p2g, "p2g")
        changed = True

    if args.g2p:
        checkpoint = replace_checkpoint(checkpoint, args.g2p, "g2p")
        changed = True        

    out = args.out if args.out else args.src_checkpoint
    if changed:
        torch.save(checkpoint, out)
    else:
        print("The original checkpoint has not been changed!")