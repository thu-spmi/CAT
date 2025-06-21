import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(description="Process nbest files and generate text mappings.")
    parser.add_argument("--f_nbest", required=True, help="Input pickle nbest file path")
    parser.add_argument("--trans_text", required=True, help="Reference transcription text file path")
    parser.add_argument("--mode", choices=["beamsearch", "sample"], default="beamsearch", help="noisy phoneme mode")
    args = parser.parse_args()

    f_path_text = args.f_nbest + "_text"
    if args.mode == "beamsearch":  # beam mode
        f_out_nbest = args.f_nbest + "_nbest"
        with open(args.f_nbest, "rb") as fi:
            _dataset = list(pickle.load(fi).items())
        with open(f_out_nbest, "w", encoding='utf-8') as fo:
            for index in range(len(_dataset)):
                okey = _dataset[index][0]
                for nid, (_score, _trans) in _dataset[index][1].items():
                    fo.write(f"{okey}\t{_trans}\n")

        text = {}
        with open(args.trans_text, "r", encoding='utf-8') as ti:
            for line in ti:
                key, value = line.strip().split(maxsplit=1)
                text[key] = value

        with open(f_path_text, 'a', encoding='utf-8') as fc:
            with open(f_out_nbest, "r", encoding='utf-8') as fi:
                uids_to_keep = [line.split()[0] for line in fi]
                for uid in uids_to_keep:
                    fc.write(f"{uid}\t{text[uid]}\n")

    else:  # sample mode
        f_out_nsamp = args.f_nbest + "_nsamp"

        with open(f_out_nsamp, "a", encoding='utf-8') as fo:
            with open(args.f_nbest, "rb") as fi:
                _dataset = list(pickle.load(fi).items())
                for index in range(len(_dataset)):
                    okey = _dataset[index][0]
                    for nid, (_score, _trans) in _dataset[index][1].items():
                        fo.write(f"{okey}\t{_trans}\n")

        text = {}
        with open(args.trans_text, "r", encoding='utf-8') as ti:
            for line in ti:
                key, value = line.strip().split(maxsplit=1)
                text[key] = value

        with open(f_path_text, 'a', encoding='utf-8') as fc:
            with open(f_out_nsamp, "r", encoding='utf-8') as fi:
                uids_to_keep = [line.split()[0] for line in fi]
                for uid in uids_to_keep:
                    fc.write(f"{uid}\t{text[uid]}\n")

if __name__ == "__main__":
    main()