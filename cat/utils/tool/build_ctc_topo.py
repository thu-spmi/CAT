"""
Author: Huahuan Zheng (maxwellzh@outlook.com)

Build CTC topo FST for denominator or decoding graph.
assume <blk> = 0 in tokenizer.

Note that the built topo is kind of special, where
... the input token is indeed real token + 1. e.g. 
... <blk> is 1 in input token, but 0 in output token.
"""
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vocab_size", type=int,
                        help="Vocabulary size (include the <blk>).")
    parser.add_argument("--n-disambig", type=int, default=0, help="Number of disambiguous symbols (include #0). "
                        "Used with --build-for-decode")
    parser.add_argument("--build-for-decode", action="store_true", default=False,
                        help="Build CTC TOPO for decoding , default: False.")
    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        sys.stderr.write("vocab size is required.\n")
        sys.exit(1)

    isdec = args.build_for_decode
    if isdec:
        assert isinstance(args.n_disambig, int)

    vocab_size = args.vocab_size
    assert vocab_size >= 2, "there should be at least one valid label besides <blk>."

    try:
        # state:0 -> state:0, input: <blk>, output: <eps>
        sys.stdout.write("0 0 1 0\n")
        sys.stdout.write("0\n")
        for i in range(1, vocab_size):
            sys.stdout.write(f"0 {i} {i+1} {i+isdec}\n")
            sys.stdout.write(f"{i} {i} {i+1} 0\n")
            sys.stdout.write(f"{i} 0 1 0\n")

        for prev_s in range(1, vocab_size):
            for next_s in range(1, vocab_size):
                if prev_s != next_s:
                    sys.stdout.write(
                        f"{prev_s} {next_s} {next_s+1} {next_s+isdec}\n"
                    )
            sys.stdout.write(f"{prev_s}\n")

        if isdec:
            # suppose all disambiguous symbols are behinded real symbols
            for i in range(vocab_size):
                for j in range(vocab_size+1, vocab_size+1+args.n_disambig):
                    sys.stdout.write(f"{i} {i} {0} {j}\n")
    except IOError:
        sys.exit(0)
