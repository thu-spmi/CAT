"""
Convert phonetic from BIPA (broad IPA) to panphon graphemes
"""

import os
import sys
import argparse
from typing import *


class PhoneticTranslator:
    def __init__(self, bipa2panphon: str = None) -> None:
        if bipa2panphon is not None:
            assert os.path.isfile(bipa2panphon), f"{bipa2panphon}"

            self._bipa = {}  # type: Dict[str, str]
            with open(bipa2panphon, "r") as fi:
                for line in fi:
                    bipa, panphon = line.strip().split(maxsplit=1)
                    self._bipa[bipa] = panphon
        else:
            self._bipa = None

    def convert_bipa2panphon(self, sym_bipa: str) -> Union[str, None]:
        """Convert phonetic from BIPA (broad IPA) to panphon graphemes"""

        if self._bipa is None:
            raise FileNotFoundError("BIPA to Panphon mapping is not configured.")

        return self._bipa.get(sym_bipa, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lexicon",
        type=str,
        help="Input lexicon file. Each line: word phone1 phone2 ...",
    )
    parser.add_argument(
        "ipa_all",
        type=str,
        help="The list contains all IPA symbols. One symbol per line.",
    )
    parser.add_argument(
        "--extend",
        type=str,
        help="Add a custom extend mapping besides the BIPA. Each line: phone ipa1 ipa2 ...",
    )
    args = parser.parse_args()

    # initialize tranlator, bipa mapping is fetched from https://github.com/cldf-clts/clts
    translator = PhoneticTranslator("local/data/bipa.txt")
    extend = {}
    if args.extend is not None:
        with open(args.extend, "r") as fi:
            for line in fi:
                if line[0] == "#":
                    # lines start with '#' would be ignored.
                    continue
                contents = line.strip().split(maxsplit=1)
                if len(contents) < 2:
                    raise ValueError(f"Invalid extend line: {line}")
                extend[contents[0]] = contents[1]

    # read ipa symbols
    IPAs = set(x for x in open(args.ipa_all, "r").read().split("\n"))

    try:
        if args.lexicon == "-":
            args.lexicon = "/dev/stdin"
        notexist = set()
        with open(args.lexicon, "r") as fi:
            for line in fi:
                contents = line.strip().split()
                if len(contents) < 2:
                    raise ValueError(f"Invalid lexicon line: {line}")
                word = contents[0]
                prons = []
                for phone in contents[1:]:
                    if phone in IPAs:
                        prons.append(phone)
                    elif (
                        converted_phone := translator.convert_bipa2panphon(phone)
                    ) is not None:
                        prons.append(converted_phone)
                    elif phone in extend:
                        prons.append(extend[phone])
                    else:
                        if phone not in notexist:
                            notexist.add(phone)
                            sys.stderr.write(f"unable to translate symbol: '{phone}'\n")

                sys.stdout.write(f"{word}\t{' '.join(prons)}\n")

    except IOError:
        pass
    sys.exit(0)
