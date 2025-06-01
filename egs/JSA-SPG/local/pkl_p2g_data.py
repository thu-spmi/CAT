# Copyright 2025 Tsinghua SPMI Lab
# Author: Author: Sardar (sar_dar@foxmail.com)

import argparse
import os
import sys
import pickle
from cat.shared.tokenizer import load
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/..'))
from utils.pipeline.common_utils import *
import kaldiio

def pack_data(
    f_scps: Union[List[str], str],
    f_labels: Union[List[str], str],
    f_out: str,
    output_tokenizer,
    input_tokenizer,
    from_given,
    from_nbest_file: str,
    isG2P
):
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
    
    # Read label files.
    twrapper_label = TextUtterancesOrdered(f_labels)
    twrapper_scp = TextUtterancesOrdered(f_scps)

    assert len(twrapper_scp) == len(twrapper_label), sfmt.error(
        "f_scp and f_label should match on the # of lines, "
        f"instead {len(twrapper_scp)} != {len(twrapper_label)}",
        pack_data,
    )
    cnt_frames = 0
    uids = []
    linfo = []
    inputs = []
    outputs = []
    cnt = 0
    if from_given:
        if isG2P:
            phone_list = output_tokenizer._units
        else:
            phone_list = input_tokenizer._units
            
    for (uid, lb), (uid1, ark) in tqdm(zip(twrapper_label, twrapper_scp), total=len(twrapper_label), leave=False
    ):
        assert uid == uid1, f"UID in label and scp files mismatch: {uid} != {uid1}"
        if lb == "":
            sfmt.warn(f"skip empty utt: {uid}", pack_data)
            continue
        
        if isG2P:
            in_put = np.asarray(input_tokenizer.encode(lb), dtype=np.int64)
            if from_given:
                out_put = np.asarray([phone_list[phone] for phone in ark.split()], dtype=np.int64)
            else:
                out_put = np.asarray(output_tokenizer.encode(lb), dtype=np.int64)
        else:
            if from_given:
                in_put = np.asarray([phone_list[phone] for phone in ark.split()], dtype=np.int64)
            else:
                in_put = np.asarray(input_tokenizer.encode(lb), dtype=np.int64)
            out_put = np.asarray(output_tokenizer.encode(lb), dtype=np.int64)
            
        if out_put.shape[0] == 0:
            continue
        outputs.append(out_put)
        linfo.append(in_put.shape[0])
        inputs.append(in_put)
        uids.append(uid)
        cnt_frames += in_put.shape[0]
        cnt += 1

    if cnt == 0:
        sys.stderr.write(sfmt.error("no qualified seq found.\n", pack_data))
        sys.exit(1)

    # in order to store labels in a ndarray,
    # first I pad all labels to the max length with -1 (this won't take many memory since labels are short compared to frames)
    # then store the length in the last place, such as
    # [0 1 2 3] -> [0 1 2 3 -1 -1 4]
    # then we can access the data via array[:array[-1]]
    cnt_tokens = sum(x.shape[0] for x in outputs)
    max_len_label = max(x.shape[0] for x in outputs)
    outputs = np.array(
        [
            np.concatenate(
                (_x, np.array([-1] * (max_len_label - _x.shape[0]) + [_x.shape[0]]))
            )
            for _x in outputs
        ]
    )
    max_len_input = max(x.shape[0] for x in inputs)
    inputs = np.array(
        [
            np.concatenate(
                (_x, np.array([-1] * (max_len_input - _x.shape[0]) + [_x.shape[0]]))
            )
            for _x in inputs
        ]
    )

    with open(f_out, "wb") as fo:
        pickle.dump(
            {
                "label": outputs,
                "linfo": np.array(linfo, dtype=np.int16),
                "arkname": inputs,
                # "arkname": np.array(arks),
                "key": uids,
            },
            fo,
        )

    cntrm = len(twrapper_label) - cnt
    if cntrm > 0:
        print(f"pack_data(): remove {cntrm} unqualified sequences.")
    print(f"# of frames: {cnt_frames} | tokens: {cnt_tokens} | seqs: {cnt}")

def pack_large_data(
    f_scps: Union[List[str], str],
    f_labels: Union[List[str], str],
    f_out: str,
    output_tokenizer,
    input_tokenizer,
    from_given,
    from_nbest_file: str,
    isG2P
):
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
    
    # Read label files.
    twrapper_label = TextUtterancesOrdered(f_labels)
    twrapper_scp = TextUtterancesOrdered(f_scps)

    if from_nbest_file:
        checkExist("f", from_nbest_file)
        with open(from_nbest_file, "rb") as f_hy:
            # type: Dict[str, Dict[int, Tuple[float, str]]]
            nbest_list = pickle.load(f_hy)

    assert len(twrapper_scp) == len(twrapper_label), sfmt.error(
        "f_scp and f_label should match on the # of lines, "
        f"instead {len(twrapper_scp)} != {len(twrapper_label)}",
        pack_data,
    )
    cnt_frames = 0

    uids = []
    in_lens = []
    out_lens = []
    inputs = []
    outputs = []
    
    cnt = 0
    if from_given or from_nbest_file:
        if isG2P:
            phone_list = output_tokenizer._units
        else:
            phone_list = input_tokenizer._units
            
    for (uid, lb), (uid1, ark) in tqdm(zip(twrapper_label, twrapper_scp), total=len(twrapper_label), leave=False
    ):
        assert uid == uid1, f"UID in label and scp files mismatch: {uid} != {uid1}"
        if lb == "":
            sfmt.warn(f"skip empty utt: {uid}", pack_data)
            continue
        if from_nbest_file:
            for n_best_id, nbest_value in nbest_list[uid].items():
                if not isG2P:
                    in_put = np.asarray([phone_list[phone] for phone in nbest_value[1].split()], dtype=np.int64)
                    out_put = np.asarray(output_tokenizer.encode(lb), dtype=np.int64)
                
                new_uid = f"{uid}_{n_best_id}"
                if out_put.shape[0] == 0 or in_put.shape[0] < out_put.shape[0] + 2:
                    continue
                # outputs.append(lb)
                # inputs.append(nbest_value[1])
                inputs.append(in_put)
                outputs.append(out_put)
                uids.append(new_uid)
                cnt_frames += in_put.shape[0]
                cnt += 1
        else:
            if not isG2P:
                if from_given:
                    in_put = np.asarray([phone_list[phone] for phone in ark.split()], dtype=np.int64)
                else:
                    in_put = np.asarray(input_tokenizer.encode(lb), dtype=np.int64)
                out_put = np.asarray(output_tokenizer.encode(lb), dtype=np.int64)
            
            if out_put.shape[0] == 0 or in_put.shape[0] < out_put.shape[0] + 2:
                continue
            inputs.append(in_put)
            outputs.append(out_put)
            uids.append(uid)
            cnt_frames += in_put.shape[0]
            cnt += 1
    

    if cnt == 0:
        sys.stderr.write(sfmt.error("no qualified seq found.\n", pack_data))
        sys.exit(1)

    cnt_tokens = sum(out_lens)

    with open(f_out, "wb") as fo:
        pickle.dump(
            {
                "label": outputs,
                "input": inputs,
                "key": uids,
            },
            fo,
        )

    cntrm = len(twrapper_label) - cnt
    if cntrm > 0:
        print(f"pack_data(): remove {cntrm} unqualified sequences.")
    print(f"# of phoneme tokens: {cnt_frames} | grapheme tokens: {cnt_tokens} | seqs: {cnt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expdir", type=str, help="text file")
    parser.add_argument("input_tokenizer_file", type=str, help="path of .tknz tokenizer file")
    parser.add_argument("--input_sqs_from_given", action='store_true', help="whether input sequence from given or from label")
    parser.add_argument("--g2p", action='store_true', help="whether a g2p task, or g2p task")
    parser.add_argument("--test", action='store_true', help="for test set just store text file. For train/dev set pickle data.")
    parser.add_argument("--from_nbest_file", type=str, default=None, help="path of nbest file for Pickle data")
    parser.add_argument("--out", type=str, help="path of output phone sequence file")
    parser.add_argument("--save2info", action='store_true', help="whether save output test file to datainfo.json.")

    args = parser.parse_args()

    working_dir = args.expdir
    checkExist("d", working_dir)
    f_hyper = os.path.join(working_dir, F_HYPER_CONFIG)
    checkExist("f", f_hyper)
    hyper_cfg = readjson(f_hyper)
    assert "data" in hyper_cfg, sfmt.missing("data", sfmt.udl(f_hyper))
    data_settings = hyper_cfg["data"]
    datainfo = readjson(F_DATAINFO)
    checkExist("f", args.input_tokenizer_file)
    input_tokenizer = load(args.input_tokenizer_file)

    if not args.test:
        print(sfmt.header("Stage 2 Pickle data"))
        fmt = sfmt(sfmt("Pickle data: ", sfmt.BOLD), sfmt.OKCYAN) + "{}\n"

        # load tokenizer from file
        assert "tokenizer" in hyper_cfg, sfmt.missing("tokenizer", sfmt.udl(f_hyper))
        assert "file" in hyper_cfg["tokenizer"], sfmt.missing(
            "file", (sfmt.udl(f_hyper), "tokenizer")
        )
        f_tokenizer = hyper_cfg["tokenizer"]["file"]
        checkExist("f", f_tokenizer)
        output_tokenizer = load(f_tokenizer)

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
            f_data = []
            
            for _set in data_settings[dataset]:
                if _set not in datainfo:
                    raise RuntimeError(
                        f"'{_set}' not found. you can configure it manually in {F_DATAINFO}"
                    )
                f_data.append(datainfo[_set])
                
            if args.from_nbest_file:
                pack_fun = pack_large_data
                from_nbest_file = args.from_nbest_file if dataset == "train" else None
                input_sqs_from_given = args.input_sqs_from_given if args.input_sqs_from_given else True
            else:
                pack_fun = pack_data
                from_nbest_file = None
                input_sqs_from_given = args.input_sqs_from_given

            pack_fun(
                [_data["scp"] for _data in f_data],
                [_data["trans"] for _data in f_data],
                f_out=os.path.join(d_pkl, dataset + ".pkl"),
                output_tokenizer=output_tokenizer,
                input_tokenizer=input_tokenizer,
                from_given=input_sqs_from_given,
                from_nbest_file=from_nbest_file,
                isG2P=args.g2p
            )
            del f_data
            
    else:
        fmt = sfmt(sfmt("Convert text data to phoneme sequence: ", sfmt.BOLD), sfmt.OKCYAN) + "{}\n"
        test_texts = []
        outs = []
        part = "trans" if args.g2p else "scp"
        prefix = "_char_id" if args.g2p else "_phn_id"
        test_sets = data_settings["test"]
        if isinstance(test_sets, str):
            test_sets = [test_sets]
        for test in test_sets:
            text = datainfo[test][part]
            test_texts.append(text)
            outs.append(text + prefix)

        if args.input_sqs_from_given:
            phone_list = input_tokenizer._units
        
        for test_text, out, set in zip(test_texts, outs, test_sets):
            with open(out, 'w', encoding='utf-8') as wf:
                with open(test_text, "r", encoding='utf-8') as f:
                    for line in f:
                        key, g_s = line.replace("\t", " ").strip().split(maxsplit=1)
                        if args.input_sqs_from_given:
                            g_s = [str(phone_list[phone]) for phone in g_s.split()]
                        else:
                            g_s = list(map(str, input_tokenizer.encode(g_s)))
                        wf.write(f"{key}\t{' '.join(g_s)}\n")
                    datainfo[set]["scp"] = out
        if args.save2info:
            dumpjson(datainfo, F_DATAINFO)