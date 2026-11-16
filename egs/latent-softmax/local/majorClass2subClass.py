import argparse
import os
import cat.shared.tokenizer as tknz
# from cat.shared.tokenizer import LexiconTokenizer
import json
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump the subclass -> major-class mapping of a tonal tokenizer."
    )
    parser.add_argument("tokenizer", type=str, help="Path to the tokenizer.")
    parser.add_argument(
        "--out",
        type=str,
        help="Path of the sub2base JSON. Default: <tokenizer dir>/sub2base.json. "
        "Use this file as `trainer:sub2base_file`, e.g. "
        "data/CS-Dialogue/baseline_sub2base.json.",
    )
    parser.add_argument(
        "--subclass-list-out",
        type=str,
        default=None,
        help="Optional path of the `subClassList` JSON, i.e. one list of subclass ids "
        "per major class, used as `trainer:latent_map` by the `latent_ctc` loss "
        "(src/latent_ctc).",
    )

    args = parser.parse_args()
    tokenizer = tknz.load(args.tokenizer)
    phone_list = tokenizer._units
    phone_list_old = phone_list.copy()
    tone_pattern = re.compile(r'\u02e7\u02e9\u02e7|\u02e5\u02e9|\u02e7\u02e5|\u02e5')
    sub2base = {}
    base2sub = {}
    for k,v in phone_list_old.items():
        match_result = tone_pattern.search(k)
        if match_result:
            # tone = match_result.group()
            major = k[:match_result.start()] + k[match_result.end():]
            if major not in phone_list:
                phone_list[major] = len(phone_list)
            sub2base[v] = phone_list[major]
            base2sub.setdefault(phone_list[major], []).append(v)
    out = args.out if args.out else os.path.join(os.path.dirname(args.tokenizer), 'sub2base.json')
    json.dump(sub2base, open(out, 'w'), indent=4)

    if args.subclass_list_out:
        # Same ordering as `TonalLexiconTokenizer.get_subClassList`, so the two
        # entry points agree.
        base_ids = sorted(base2sub)
        json.dump(
            [base2sub[b] for b in base_ids],
            open(args.subclass_list_out, 'w'),
            indent=4,
        )
        print(f"### wrote subClassList to {args.subclass_list_out}")

    for k,v in sub2base.items():
        print(f"{tokenizer.decode([k])}: {tokenizer.decode([v])}")
