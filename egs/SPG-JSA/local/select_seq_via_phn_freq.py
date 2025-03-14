# Copyright 2025 Tsinghua SPMI Lab
# Author: Sardar (sar_dar@foxmail.com)

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str, help="path of data dir"
    )
    parser.add_argument(
        "--data_duration", type=int, default=0, help="select num. of seconds data"
    )
    parser.add_argument(
        "--num_utt", type=int, default=0, help="select num. of sentence"
    )
    parser.add_argument(
        "--special_phn_list", type=str, default=None, 
        help="the file path that contains the list of phonemes for which you need to ensure the quantity when making a selection."
    )
    parser.add_argument(
        "--out", type=str, default=None, help="output path of checkpoint file"
    )
    args = parser.parse_args()

    assert args.data_duration or args.num_utt, "must provide 'data_duration' or 'num_utt'"
    
    if args.num_utt:
        phn_file = os.path.join(args.data_dir, "text_phn")
        assert os.path.isfile(phn_file), "this script require text_phn."

    if args.data_duration:
        duration_file = os.path.join(args.data_dir, "utt2dur")
        assert os.path.isfile(duration_file), "this script require utt2dur for calculate total duration."
        duration_dict = {}
        with open(duration_file, "r") as f:
            for line in f:
                uid, duration = line.strip().split()
                duration_dict[uid] = float(duration)

    # given_phn = ["ɖ͡ʐ", "d͡ʑ", "t͡ɕ", "ʈ͡ʂ"]    # pl   unseen phones
    # given_phn = ["ɖ͡ʐ", "d͡ʑ", "t͡ɕ", "ʈ͡ʂ", "ŋ", "d͡z", "ʑ", "f"]    # pl   added low freqency phones appear in train_pl
    # given_phn = ["q", "ʃ", "ɣ", "z", "v", "x"]     # id
    given_phn = []
    if args.special_phn_list:
        assert os.path.isfile(args.special_phn_list), f"{args.special_phn_list} not a valid file."
        with open(args.special_phn_list, "r", encoding='utf-8') as f:
            for line in f:
                phn = line.strip().split(maxsplit=1)[0]
                given_phn.append(phn)
    
    freq_dict = {}
    freq_list = {}
    seq_dict = {}
    overall_phone_freq = {}
    with open(phn_file, "r", encoding='utf-8') as f:
        for line in f:
            uid, seq = line.strip().split(maxsplit=1)
            freq = 0
            tmp = []
            phone_freq = {}
            for phn in seq.split():
                if phn in given_phn:
                    freq += 1
                phone_freq[phn] = phone_freq.get(phn, 0) + 1
            seq_dict[uid] = seq
            freq_list[uid] = freq
            freq_dict[uid] = phone_freq
            for p, f in phone_freq.items():
                overall_phone_freq[p] = overall_phone_freq.get(p, 0) + f

    sorted_freq_list = sorted(freq_list.items(), key=lambda item: item[1], reverse=True)

    out_file = args.out if args.out else os.path.join(args.data_dir, "selected_uids")
    acc_duration = 0
    n = 0
    selected_uid = set()
    selected_seq = set()
    
    selected = False
    not_selected = set()
    selected_phone_freq = {}
    for i in overall_phone_freq.keys():
        selected_phone_freq[i] = 0
    
    if args.special_phn_list:
        target_phone = given_phn[0]
    else:
       target_phone =  min(overall_phone_freq, key=overall_phone_freq.get)

    with open(out_file, "w", encoding='utf-8') as wf:
        while True:
            for uid, freq in sorted_freq_list:
                if freq_dict[uid].get(target_phone, 0) == 0:
                    continue
                if uid in selected_uid:                # avoid uid repetition
                    continue
                if seq_dict[uid] in selected_seq:      # avoid text data repetition
                    continue
                # selected
                selected = True
                wf.write(f"{uid}\n")
                if args.data_duration:
                    acc_duration += duration_dict[uid]
                selected_uid.add(uid)
                selected_seq.add(seq_dict[uid])
                n += 1
                for p, f in freq_dict[uid].items():
                    selected_phone_freq[p] = selected_phone_freq.get(p, 0) + f
                break
            if not selected:
                not_selected.add(target_phone)
            
            if n == args.num_utt:
                print(f"Selection over. {n} utterances are selected")
                break
            if args.data_duration and acc_duration > args.data_duration:
                print(f"Selection over. {acc_duration} second data are selected")
                break
            sorted_phone_freq = sorted(selected_phone_freq.items(), key=lambda item: item[1])
            
            # guarantee have enough phone
            for p, f in sorted_phone_freq:
                target_phone = p
                if target_phone not in not_selected:
                    break
            selected = False
            # print(f"{n}: {target_phone} selected {selected_phone_freq.get(target_phone, 0)} times over {overall_phone_freq.get(target_phone, 0)}.")
