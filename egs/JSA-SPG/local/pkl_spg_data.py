# Copyright 2025 Tsinghua SPMI Lab
# Author: Sardar (sar_dar@foxmail.com)

import argparse
import os
import sys
import pickle
from cat.shared.tokenizer import AbsTokenizer, LexiconTokenizer, load
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/..'))
from utils.pipeline.common_utils import *
import kaldiio

def pack_data(
    f_scps: Union[List[str], str],
    f_labels: Union[List[str], str],
    f_out: str,
    tokenizer1: AbsTokenizer,
    tokenizer2: AbsTokenizer,
    filter: Optional[str] = None,
):
    """Parsing audio feature and text label into pickle file.

    Args:
        f_scps   (str, list): Kaldi-like-style .scp file(s).
        f_labels (str, list): Pure text file(s) include utterance id and sentence labels. Split by space.
        f_out    (str): Ouput pickle file location.
        tokenizer1 (AbsTokenizer): BPE tokenizer for P2G.
        tokenizer2 (AbsTokenizer): Character tokenizer for G2P in the training, Phoneme tokenizer for S2P in the evaluating.
        filter (str, optional): identifier for filtering out seqs with unqualified length.
            such as '100:2000' means remove those whose length is shorter than 100 or longer than 2000. Default: None
    """
    import kaldiio
    import numpy as np
    from tqdm import tqdm

    if os.path.isfile(f_out):
        sys.stderr.write(
            sfmt.warn(
                f"file exist: {sfmt.udl(f_out)}, "
                "rm it if you want to update the data.\n",
                pack_data,
            )
        )
        return

    if isinstance(f_scps, str):
        f_scps = [f_scps]
    if isinstance(f_labels, str):
        f_labels = [f_labels]

    checkExist("f", f_scps + f_labels)
    checkExist("d", os.path.dirname(f_out))

    l_min = 1
    l_max = float("inf")
    if filter is not None:
        assert ":" in filter, sfmt.error(f"invalid filter format {filter}", pack_data)
        l_bound, u_bound = (i for i in filter.split(":"))
        if l_bound != "":
            l_min = int(l_bound)
        if u_bound != "":
            l_max = int(u_bound)
    
    # Read label files.
    twrapper_label = TextUtterancesOrdered(f_labels)
    twrapper_scp = TextUtterancesOrdered(f_scps)
    assert len(twrapper_scp) == len(twrapper_label), sfmt.error(
        "f_scp and f_label should match on the # of lines, "
        f"instead {len(twrapper_scp)} != {len(twrapper_label)}",
        pack_data,
    )
    cnt_frames = 0

    f_opened = {}
    cnt_frames = 0
    linfo = np.empty(len(twrapper_scp), dtype=np.int64)
    uids = []
    arks = []
    labels1 = []
    labels2 = []
    cnt = 0
    for (uid, lb), (uid1, ark) in tqdm(
        zip(twrapper_label, twrapper_scp), total=len(twrapper_label), leave=False
    ):
        assert uid == uid1, f"UID in label and scp files mismatch: {uid} != {uid1}"
        if lb == "":
            sfmt.warn(f"skip empty utt: {uid}", pack_data)
            continue
        
        mat = kaldiio.load_mat(ark, fd_dict=f_opened)  # type:np.ndarray
        if mat.shape[0] < l_min or mat.shape[0] > l_max:
            continue

        lb1 = np.asarray(tokenizer1.encode(lb), dtype=np.int64)
        lb2 = np.asarray(tokenizer2.encode(lb), dtype=np.int64)

        if lb1.shape[0] == 0 or lb2.shape[0] == 0:
            continue

        labels1.append(lb1)
        labels2.append(lb2)
        linfo[cnt] = mat.shape[0]
        uids.append(uid)
        arks.append(ark)
        cnt_frames += mat.shape[0]
        cnt += 1
    
    for f in f_opened.values():
        f.close()

    if cnt == 0:
        sys.stderr.write(sfmt.error("no qualified seq found.\n", pack_data))
        sys.exit(1)

    # in order to store labels in a ndarray,
    # first I pad all labels to the max length with -1 (this won't take many memory since labels are short compared to frames)
    # then store the length in the last place, such as
    # [0 1 2 3] -> [0 1 2 3 -1 -1 4]
    # then we can access the data via array[:array[-1]]
    cnt_tokens1 = sum(x.shape[0] for x in labels1)
    max_len_label1 = max(x.shape[0] for x in labels1)
    labels1 = np.array(
        [
            np.concatenate(
                (_x, np.array([-1] * (max_len_label1 - _x.shape[0]) + [_x.shape[0]]))
            )
            for _x in labels1
        ]
    )
    cnt_tokens2 = sum(x.shape[0] for x in labels2)
    max_len_label2 = max(x.shape[0] for x in labels2)
    labels2 = np.array(
        [
            np.concatenate(
                (_x, np.array([-1] * (max_len_label2 - _x.shape[0]) + [_x.shape[0]]))
            )
            for _x in labels2
        ]
    )

    with open(f_out, "wb") as fo:
        pickle.dump(
            {
                "label1": labels1,
                "label2": labels2,
                "linfo": linfo[:cnt],
                "arkname": np.array(arks),
                "key": np.array(uids),
            },
            fo,
        )

    cntrm = len(twrapper_label) - cnt
    if cntrm > 0:
        print(f"pack_data(): remove {cntrm} unqualified sequences.")
    text2 = "Phoneme" if isinstance(tokenizer2, LexiconTokenizer) else "Character"
    print(f"# of frames: {cnt_frames} | BPE tokens: {cnt_tokens1} | {text2} tokens: {cnt_tokens2} | seqs: {cnt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expdir", type=str, help="text file")
    args = parser.parse_args()

    cwd = os.getcwd()
    working_dir = args.expdir
    checkExist("d", working_dir)
    f_hyper = os.path.join(working_dir, F_HYPER_CONFIG)
    checkExist("f", f_hyper)
    hyper_cfg = readjson(f_hyper)
    assert "data" in hyper_cfg, sfmt.missing("data", sfmt.udl(f_hyper))
    data_settings = hyper_cfg["data"]
    datainfo = readjson(F_DATAINFO)

    print(sfmt.header("Stage 2 Pickle data"))
    fmt = sfmt(sfmt("Pickle data: ", sfmt.BOLD), sfmt.OKCYAN) + "{}\n"

    # load tokenizer from file
    assert "tokenizer" in hyper_cfg, sfmt.missing("tokenizer", sfmt.udl(f_hyper))
    assert "file" in hyper_cfg["tokenizer"], sfmt.missing(
        "file", (sfmt.udl(f_hyper), "tokenizer")
    )
    bpe_tokenizer_file = hyper_cfg["tokenizer"]["file"]
    checkExist("f", bpe_tokenizer_file)
    bpe_tokenizer = load(bpe_tokenizer_file)

    phone_tokenizer_file = hyper_cfg['tokenizer']['phone_tokenizer']
    checkExist("f", phone_tokenizer_file)
    phone_tokenizer = load(phone_tokenizer_file)
    char_tokenizer_file = hyper_cfg['tokenizer']['char_tokenizer']
    checkExist("f", char_tokenizer_file)
    char_tokenizer = load(char_tokenizer_file)

    d_pkl = os.path.join(working_dir, "pkl")
    os.makedirs(d_pkl, exist_ok=True)
    
    for dataset in ["dev", "train"]:
        if dataset not in data_settings:
            sys.stderr.write(
                sfmt.missing(dataset, "data", raiseerror=False) + ", skip.\n"
            )
            continue
        if isinstance(data_settings[dataset], str):
            data_settings[dataset] = [data_settings[dataset]]
        
        # separate packing
        if dataset == "train":
            trset_list = []
        for _set in data_settings[dataset]:
            if _set not in datainfo:
                raise RuntimeError(
                    f"'{_set}' not found. you can configure it manually in {F_DATAINFO}"
                )
        
            f_out=os.path.join(d_pkl, _set + ".pkl") if dataset == "train" else os.path.join(d_pkl, dataset + ".pkl")
            pack_data(
                [datainfo[_set]["scp"]],
                [datainfo[_set]["trans"]],
                f_out=f_out,
                tokenizer1=bpe_tokenizer,
                tokenizer2=char_tokenizer if dataset == "train" else phone_tokenizer,
            )

            if dataset == "train":
                trset_list.append(f_out)

    hyper_cfg["train"]["option"]["trset"] = trset_list
    dumpjson(hyper_cfg, f_hyper)