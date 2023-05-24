"""
The Transformer decoder impl. was originally named as 'trans' in CausalTransformer,
which is now replaced by 'clm' to be consistent with other CLMs.

NOTE: only CausalTransformer lm is affected.

Usage:
    python utils/compat/update_transformer_lm.py /path/to/checkpoint.pt
"""


import torch
import sys
import os
from collections import OrderedDict

if __name__ == "__main__":
    if len(sys.argv[1:]) != 1:
        raise RuntimeError("Require one argument to specify the checkpoint.")

    file = sys.argv[1]
    assert os.path.isfile(file), file

    check = torch.load(file, "cpu")
    m = check["model"]
    newdict = OrderedDict()

    for k, v in m.items():
        newdict[k.replace(".trans.", ".clm.")] = v

    for prefix in ["lm", "module.lm"]:
        orin_name = prefix + ".embedding.weight"
        if orin_name in newdict:
            del newdict[orin_name]
            newdict[prefix + ".clm.wte.weight"] = m[orin_name]
    check["model"] = newdict
    torch.save(check, file)
