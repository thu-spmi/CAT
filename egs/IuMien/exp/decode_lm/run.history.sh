# you need to run python local/get_wordlist.py to get word_list if you don't have word list
# train tokenizer and pickle data
utils/pipeline/ngram.sh exp/decode_lm -o 4 --arpa --output exp2/decode_lm/4gram.arpa --sta 1 --sto 2
# train lm
utils/pipeline/ngram.sh exp/decode_lm -o 4 --arpa --output exp2/decode_lm/4gram.arpa --sta 3 --sto 3
# test lm
# utils/pipeline/ngram.sh exp2/decode_lm -o 4 --arpa --output exp2/decode_lm/4gram.arpa --sta 4 --sto 4