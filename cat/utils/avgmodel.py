#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# https://github.com/pytorch/fairseq/blob/main/scripts/average_checkpoints.py
"""
usage:
    python utils/avgmodel.py -h
"""

import os
import argparse
import collections
from typing import Literal, List

import torch


def average_checkpoints(inputs: List[str]):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for fpath in inputs:
        if not os.path.isfile(fpath):
            raise RuntimeError(f"{fpath} is not a checkpoint file.")

        state = torch.load(fpath, map_location='cpu')

        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state["model"]

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(fpath, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] = torch.div(
                averaged_params[k], num_models, rounding_mode='floor')
    new_state["model"] = averaged_params
    return new_state


def select_checkpoint(f_checklist: str, n: int = -1, mode: Literal['best', 'last', 'slicing'] = 'best', _slicing: tuple = None):
    from cat.shared.manager import CheckManager
    from cat.shared._constants import F_CHECKPOINT_LIST
    if os.path.isfile(f_checklist):
        cm = CheckManager(f_checklist)
    elif os.path.isdir(f_checklist):
        cm = CheckManager(os.path.join(f_checklist, F_CHECKPOINT_LIST))
    else:
        raise RuntimeError(f"'{f_checklist}' is neither a file nor a folder.")

    checklist = [(path, check['metric']) for path, check in cm.content.items()]

    if mode == 'best':
        assert n > 0
        assert n <= len(checklist)
        checklist = sorted(checklist, key=lambda x: x[1])[:n]
    elif mode == 'last':
        assert n > 0
        assert n <= len(checklist)
        checklist = checklist[-n:]
    else:
        assert _slicing is not None
        checklist = checklist[_slicing[0]:_slicing[1]]
    return [path for path, _ in checklist]


def main():
    parser = argparse.ArgumentParser(
        description="Tool to average the params of input checkpoints to "
        "produce a new checkpoint",
    )

    parser.add_argument('input', nargs='+', metavar='CHECK',
                        help='Input checkpoint(s) or the checkpoint list file.')
    parser.add_argument('--output', type=str, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.')
    parser.add_argument("--num-best", type=int, metavar='N',
                        help='Average the best <N> checkpoints. If set, <CHECK> must be the checkpoint list file.')
    parser.add_argument("--slicing-select", type=str,
                        help="Manually select the checkpoint(s) to be averaged in python slicing way "
                        "(e.g. '-10:' means the last ten checkpoints). "
                        "If there are negative numbers use --slicing-select=xxx:xxx "
                        "If set, --input MUST be the checkpoint list file. This is conflict with --num-best")
    parser.add_argument("--keep-model-only", action="store_true", default=False,
                        help="Remove states other than model parameters (such as scheduler / optimizer) to reduce the output file size.")

    args = parser.parse_args()
    from cat.shared.manager import CheckManager

    if args.num_best is None and args.slicing_select is None:
        notfound = []
        for f in args.input:
            if not os.path.isfile(f):
                notfound.append(f)
        if len(notfound) > 0:
            raise FileNotFoundError("\n{}".format('\n'.join(notfound)))

        assert args.output is not None
        pathlist = args.input
    else:
        assert len(args.input) == 1, \
            "when doing averaging with checkpoint list file, " \
            f"only one input is accepted. However, given input is {args.input}"
        f_checklist = args.input[0]
        try:
            CheckManager(f_checklist)
        except Exception as e:
            print(str(e))
            raise ValueError(
                "Seems like the CHECK is not a valid checkpoint list file.")

        if args.num_best is not None:
            pathlist = select_checkpoint(
                f_checklist, args.num_best, mode='best')
            if args.output is None:
                args.output = os.path.join(os.path.dirname(
                    f_checklist), f"avg_best_{args.num_best}.pt")
        else:
            assert ':' in args.slicing_select, f"invalid slicing format: '{args.slicing_select}'"
            slicing = args.slicing_select.split(':')
            assert len(slicing) == 2
            if args.output is None:
                args.output = os.path.join(
                    os.path.dirname(f_checklist),
                    f"avg_slice_{args.slicing_select}.pt"
                )

            if slicing[0] == '':
                slicing[0] = '0'
            if slicing[1] == '':
                slicing[1] = str(2**32-1)

            slicing = [int(x) for x in slicing]
            pathlist = select_checkpoint(
                f_checklist, mode='slicing', _slicing=slicing)

    assert len(pathlist) > 0
    if args.output is None:

        pass

    print("Averaging checkpoints:\n"+'\n'.join(pathlist))
    new_state = average_checkpoints(pathlist)
    if args.keep_model_only:
        for k in list(new_state.keys()):
            if k != 'model':
                del new_state[k]
    torch.save(new_state, args.output)
    print("Finished writing averaged checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()
