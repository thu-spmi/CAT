"""
In previous version, the encoder in CTC trainer is named as 'am', thus the 
names of parameters are like 'module.am.xxx'

Now the 'am' is replaced to 'encoder' to be consistent with other non-CTC models.
So we have to replace the 'am' to 'encoder' to allow loading
models from previous checkpoints.

Usage:
    python utils/compat/repl_am_to_encoder.py /path/to/checkpoint.pt
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
        newdict[k.replace(".am.", ".encoder.")] = v

    check["model"] = newdict
    torch.save(check, file)
