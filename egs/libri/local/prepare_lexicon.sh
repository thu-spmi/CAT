#!/bin/bash

mkdir -p data/local
[ ! -f data/local/librispeech-lexicon.txt ] &&
    wget https://www.openslr.org/resources/11/librispeech-lexicon.txt -P data/local

echo "$0 done."
exit 0
