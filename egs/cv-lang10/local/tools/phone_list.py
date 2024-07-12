import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lexicon", type=str, help="lexicon file")
    parser.add_argument("--out", type=str, help="path of output phone list file")
    args = parser.parse_args()

    assert os.path.isfile(args.lexicon), f"phone_list={args.lexicon} is not a valid file."

    phone_list = set()

    with open(args.lexicon, "r", encoding="utf-8") as f:
        for line in f:
            phone_seq = line.strip().split('\t', maxsplit=1)[1]
            phone_list.update(phone_seq.split())
    
    out = args.out if args.out else os.path.join(os.path.dirname(args.lexicon), "phone_list.txt")

    with open(out, "w", encoding="utf-8") as wf:
        for phone in phone_list:
            if phone != ' ':
                wf.write(phone + "\n")

    