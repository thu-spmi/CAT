
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("word_list", type=str, help="text file")
    parser.add_argument("--out", type=str, help="path of output char list file")
    args = parser.parse_args()

    assert os.path.isfile(args.word_list), f"word_list={args.word_list} is not a valid file."

    char_list = set()

    with open(args.word_list, "r", encoding="utf-8") as f:
        for line in f:
            char_list.update(list(line.strip()))
    
    out = args.out if args.out else os.path.join(os.path.dirname(args.word_list), "char_list.txt")

    with open(out, "w", encoding="utf-8") as wf:
        for char in char_list:
            wf.write(char + "\n")

    
