"""
A simple script for cleaning the experiment folder.
"""


import re
import os
import sys
import shutil
from datetime import datetime
from typing import *

# fmt:off
try:
    import utils.pipeline
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/..'))
from utils.pipeline._constants import *
# fmt:on


def rm(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        if os.path.islink(path):
            os.unlink(path)
        else:
            shutil.rmtree(path)
    else:
        return


class JoinPath:
    def __init__(self, parent: str) -> None:
        self._parent = parent

    def __call__(self, name) -> Any:
        return os.path.join(self._parent, name)


if __name__ == "__main__":
    if sys.argv[1:] == []:
        print(
            "Usage: \n"
            f"    python {sys.argv[0]} DIR\n\n"
            "Used for clean experiment folder."
        )

    path = sys.argv[1]
    if not os.path.isdir(path):
        print(
            f"Folder: '{path}' is empty / not a directory."
        )
        sys.exit(0)

    join = JoinPath(path)

    res = []        # type: List[Tuple[str, str]]
    # checkdir
    if os.path.isdir(d := join(D_CHECKPOINT)):
        res.append(
            (d, "checkpoint folder")
        )

    # decode dir
    if os.path.isdir(d := join(D_INFER)):
        res.append(
            (d, "inference folder")
        )

    # log dir
    if os.path.isdir(d := join(D_LOG)):
        res.append(
            (d, "log folder")
        )

    # data foler
    if os.path.isdir(d := join('pkl')):
        res.append(
            (d, "data folder")
        )

    # temp folder
    if os.path.isdir(d := join(D_TMP)):
        res.append(
            (d, "temp folder")
        )

    # monitor fig
    if os.path.isfile(f := join(F_MONITOR_FIG)):
        res.append(
            (f, "monitor fig")
        )

    # readme
    if os.path.isfile(f := join(F_TRAINING_INFO)):
        res.append(
            (f, "readme")
        )

    # tokenizer
    if os.path.isfile(f := join(F_TOKENIZER)):
        res.append(
            (f, "tokenizer")
        )

    try:
        if len(res) > 0:
            ali = max(len(s) for _, s in res)+1
            print(
                f"Items found in {path}:\n" +
                '\n'.join([
                    f"  [{i+1}] {s+' '*(ali-len(s))}: {p}"
                    for i, (p, s) in enumerate(res)
                ])
            )
            s_indices = input("index to be removed > ")
            s_indices = re.sub(r'\s+', ' ', s_indices)
            if s_indices == 'all':
                cf_str = f"confirm {path}"
                response = input(f"Ensure by typing > {cf_str} <: ")
                if response == cf_str:
                    for p, _ in res:
                        rm(p)
                    print("Done.")
                else:
                    print("Not remove.")
            else:
                indices = []
                for i in s_indices:
                    try:
                        i = int(i)
                    except:
                        continue
                    if i > 0 and i <= len(res):
                        indices.append(i)
                if indices == []:
                    print("no valid index.")
                    sys.exit(0)
                response = input(f"valid indices: {indices}. Confirm [y/N]: ")
                if response.lower() == 'y':
                    for idx in indices:
                        rm(res[idx-1][0])
                    print("Done.")
                else:
                    print("Not remove.")

        else:
            print("Nothing found.")
    except KeyboardInterrupt:
        print("")
