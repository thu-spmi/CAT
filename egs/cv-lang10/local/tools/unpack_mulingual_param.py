"""
Get params from a model pretrained with multilingual.

flatphone:
    encoder -> linear classifier
    unpack monolingual params from the linear classifier
    'module.encoder.classifier.weight' : (C, H)
    'module.encoder.classifier.bias'   : (C, )

joinap lienar:
    encoder -> A P matrix
    unpack monolingual params from A P matrix
    'module.encoder.A.weight'   : (H, D_ipa)
    'module.encoder.A.bias'     : (H, )

joinap non-lienar:
    encoder -> A1 A2 P matrix
    unpack monolingual params from A1 A2 P matrix
    'module.encoder.A1.weight'  : (H1, D_ipa)
    'module.encoder.A1.bias'    : (H1, )
    'module.encoder.A2.weight'  : (H, H1)
    'module.encoder.A2.bias'    : (H, )
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from collections import OrderedDict
from typing import *

from cat.shared.tokenizer import load, LexiconTokenizer, JiebaComposeLexiconTokenizer


def unpack_param(
    model: OrderedDict[str, torch.Tensor],
    mapping_list: torch.LongTensor,
    mode: Literal["flatphone", "joinap-linear", "joinap-nonlinear"] = "flatphone",
) -> OrderedDict:
    # a shallow copy is OK.
    m_updated = model.copy()
    mapping = torch.LongTensor(mapping_list)
    old_num_classes = m_updated["module.encoder.classifier.weight"].size(0)
    new_num_classes = mapping.size(0)

    
    indices = torch.nonzero(mapping == -1).squeeze()
    if mode == "flatphone":
        m_updated["module.encoder.classifier.weight"] = m_updated[
            "module.encoder.classifier.weight"
        ][mapping]
        m_updated["module.encoder.classifier.bias"] = m_updated[
            "module.encoder.classifier.bias"
        ][mapping]
        
        if old_num_classes >= new_num_classes:
            if len(np.array(indices)):
                sys.stderr.write(f"WARNING: The multilingual dict cannot cover the monolingual dict\n")
                new_linear = nn.Linear(m_updated["module.encoder.classifier.weight"].shape[1], len(mapping))
                for ind in indices:
                    m_updated["module.encoder.classifier.weight"][ind] = new_linear.weight[ind]
                    m_updated["module.encoder.classifier.bias"][ind] = new_linear.bias[ind]
            pass
    elif mode == "joinap-linear":
        pass
    elif mode == "joinap-nonlinear":
        pass
    else:
        raise ValueError(f"'{mode}' is not supported.")
    return m_updated


def extract_lexicon_units(
    tknz: Union[LexiconTokenizer, JiebaComposeLexiconTokenizer]
) -> Dict[str, int]:
    assert isinstance(
        tknz, (LexiconTokenizer, JiebaComposeLexiconTokenizer)
    ), f"Unsupport tokenizer: {tknz.__class__}"

    if isinstance(tknz, JiebaComposeLexiconTokenizer):
        tknz = tknz._w2p_tokenizer

    return tknz._units


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mul_tokenizer", type=str, help="Path to the multilingual tokenizer."
    )
    parser.add_argument(
        "mono_tokenizer", type=str, help="Path to the monolingual tokenizer."
    )
    parser.add_argument(
        "src_checkpoint",
        type=str,
        help="Path to the multilingual pretrained checkpoint.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["flatphone", "joinap-linear", "joinap-nonlinear"],
        default="flatphone",
        help="Unpack mode. Default: flatphone",
    )
    args = parser.parse_args()

    for x in ["mul_tokenizer", "mono_tokenizer", "src_checkpoint"]:
        assert os.path.isfile(getattr(args, x)), x

    units_mul = extract_lexicon_units(load(args.mul_tokenizer))
    units_mono = extract_lexicon_units(load(args.mono_tokenizer))

    try:
        # sorted() here is not necessary for common cases.
        matching = []
        cnt_uncover = 0
        for u, _ in sorted(units_mono.items(), key=lambda x: x[1]):
            if u in units_mul:
              matching.append(units_mul[u])
            else:
              matching.append(-1)  
              cnt_uncover += 1

        
    except KeyError as e:
        sys.stderr.write(f"{e}\n")
        sys.stderr.write(
            "The multilingual tokenizer cannot cover the full list of the monolingual tokenizer.\n"
        )
        sys.exit(1)

    checkpoint = torch.load(args.src_checkpoint, "cpu")
    assert "model" in checkpoint
    checkpoint["model"] = unpack_param(checkpoint["model"], matching, mode=args.mode)
    print("Total phonemes of the language:", len(matching))
    print("Uncovered phonemes:", cnt_uncover)

    torch.save(checkpoint, args.output)
