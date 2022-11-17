"""
Prepare Chinese character index to syllable index mapping.

Author: Huahuan Zheng
"""
import sys
import pickle
import numpy as np


if __name__ == "__main__":
    if len(sys.argv[1:]) == 0:
        print(f"Usage: python {sys.argv[0]} /path/to/tokenizer")
        sys.exit(1)

    try:
        import pypinyin
    except ModuleNotFoundError:
        print("module:pypinyin not found. install with:")
        print("pip install pypinyin")
        sys.exit(1)

    from pypinyin import pinyin, Style
    import cat.shared.tokenizer as tknz

    tokenizer = tknz.load(sys.argv[1])

    char2pinyin = {}
    for id, chr in tokenizer.dump_vocab().items():
        if chr == '<s>' or chr == '<unk>':
            continue
        _pron = pinyin(chr, style=Style.TONE3)[0][0]
        char2pinyin[id] = _pron

    # keep 0 for <s>/<blk>, 1 for <unk>
    prons = {
        phn: (idx+2)
        for idx, phn in enumerate(sorted(set(char2pinyin.values())))
    }

    char2pinyin = [(cid, prons[phn]) for cid, phn in char2pinyin.items()]
    char2pinyin += [(0, 0), (1, 1)]
    char2pinyin = sorted(char2pinyin, key=lambda x: x[0])
    char2pinyin = list(zip(*char2pinyin))[1]

    output = 'char2syllable.pkl'
    with open(output, 'wb') as fob:
        pickle.dump({
            'converter': np.asarray(char2pinyin, dtype=np.int64),
            'syllable': prons,
            'num_syllables': len(prons)+2
        }, fob)
    print(f"Syllable converting matrix saved at: {output}")
