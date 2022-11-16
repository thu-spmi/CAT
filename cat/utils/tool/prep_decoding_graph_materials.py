"""
Author: Huahuan Zheng (maxwellzh@outlook.com)

Prepare the lexiconp.txt units.txt words.txt for decoding graph construction.
"""

import os
import sys
import argparse

from typing import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizerT", type=str,
                        help="Path to the T tokenizer.")
    parser.add_argument("tokenizerG", type=str,
                        help="Path to the G tokenizer.")
    parser.add_argument("out_dir", type=str,
                        help="Output directory, where all files will be store.")
    args = parser.parse_args()
    d_out = args.out_dir

    assert os.path.isfile(
        args.tokenizerT), f"'{args.tokenizerT}' is not a valid file."
    assert os.path.isfile(
        args.tokenizerG), f"'{args.tokenizerG}' is not a valid file."

    import cat.shared.tokenizer as tknz
    tokenizerG = tknz.load(args.tokenizerG)
    vocabulary = {c: i for i, c in tokenizerG.dump_vocab().items()}
    # NOTE: r_words.txt is the text to index mapping, useful at decoding.
    tokenizerG.dump_vocab(os.path.join(d_out, "r_words.txt"))
    if '<s>' in vocabulary:
        del vocabulary['<s>']
    if '</s>' in vocabulary:
        del vocabulary['</s>']

    if args.tokenizerT == args.tokenizerG:
        tokenizerT = tokenizerG
    else:
        tokenizerT = tknz.load(args.tokenizerT)
    del tokenizerG

    if isinstance(tokenizerT, (tknz.LexiconTokenizer, tknz.JiebaComposeLexiconTokenizer)):
        if isinstance(tokenizerT, tknz.JiebaComposeLexiconTokenizer):
            tokenizerT = tokenizerT._w2p_tokenizer
        word2unit = {k: v for k, v in tokenizerT._w2pid.items(
        ) if k in vocabulary}   # type: Dict[str, Tuple[int,]]
        unit2index = tokenizerT._units  # type: Dict[str, int]
        reversed_units = {index: unit for unit, index in unit2index.items()}
        lexicon = [
            (vocabulary[word], ' '.join(reversed_units[i] for i in units))
            for word, units in word2unit.items()
        ]    # type: List[Tuple[int, str]]

        # here we use i+1 for shifting the position 0 for <eps>
        units = [(c, i+1) for c, i in unit2index.items()]
        units[0] = ('<blk>', 1)
        del reversed_units
    else:
        # must support dump_vocab method
        index2unit = {i: w for i, w in tokenizerT.dump_vocab().items()
                      if w in vocabulary}
        lexicon = [
            (vocabulary[w], w)
            for w in index2unit.values()
        ]
        units = [('blk', 1)] + [(c, i+1) for i, c in index2unit.items()]
        del index2unit

    if len(lexicon) < 2:
        sys.stderr.write(
            "The tokenizerG cannot cover any of the word in tokenizerT.\n")
        sys.exit(1)

    del vocabulary
    del tokenizerT

    file = os.path.join(d_out, 'lexiconp.txt')
    try:
        # 1.0 represents the prob of the word.
        with open(file, 'w') as fo:
            fo.write(f"<unk> 1.0 {lexicon[0][1]}\n")
            for w, s in lexicon[1:]:
                fo.write(f"{w} 1.0 {s}\n")
        sys.stderr.write(f"Done {file}\n")
    except Exception as e:
        os.remove(file)
        raise Exception(str(e))

    file = os.path.join(d_out, 'units.txt')
    try:
        with open(file, 'w') as fo:
            for c, i in units:
                fo.write(f"{c} {i}\n")
        sys.stderr.write(f"Done {file}\n")
    except Exception as e:
        os.remove(file)
        raise Exception(str(e))

    file = os.path.join(d_out, 'words.txt')
    try:
        with open(file, 'w') as fo:
            fo.write("<eps> 0\n")
            # NOTE: in lexicon_disambig, <s> and </s> are removed, this doesn't matter.
            # in following steps, we will use arpa2fst --disambig-symbol=#0
            fo.write("<unk> 1\n")
            for w, _ in lexicon[1:]:
                fo.write(f"{w} {w}\n")
            offset = max(map(lambda x: x[0], lexicon[1:]))
            fo.write(f"#0 {offset+1}\n")
            fo.write(f"<s> {offset+2}\n")
            fo.write(f"</s> {offset+3}\n")
        sys.stderr.write(f"Done {file}\n")
    except Exception as e:
        os.remove(file)
        raise Exception(str(e))
