# prepare denominator LM (for CRF training.)
# author: Huahuan Zheng
set -e
<<"PARSER"
("r_specifier", type=str, help="Input file (raw text with ids). Use /dev/stdin if needed.")
("w_specifier", type=str, help="Output den-lm file.")
("-tokenizer", type=str, help="Path to the tokenizer file.")
("-kaldi-root", type=str, help="Path to kaldi folder. Not required if $KALDI_ROOT is set.")
("-ngram-order", type=int, default=4, help="N-gram LM order.")
("-no-prune-ngram-order", type=int, default=3, help="Passed to chain-est-phone-lm tool.")
PARSER
eval $(python utils/parseopt.py $0 $*)

if [ $kaldi_root != "None" ]; then
    export KALDI_ROOT=$kaldi_root
else
    [ -z $KALDI_ROOT ] && (
        echo "\$KALDI_ROOT is not specified."
        exit 1
    )
fi

[ ! -f $tokenizer ] && (
    echo "No such file: '$tokenizer'"
    exit 1
)

! [[ -d $KALDI_ROOT && -d $KALDI_ROOT/egs/wsj/s5 ]] && (
    echo "kaldi tool at '$KALDI_ROOT' not installed."
    exit 1
)
cd $KALDI_ROOT/egs/wsj/s5 && . ./path.sh
cd - >/dev/null

python utils/data/corpus2index.py $r_specifier -t --tokenizer=$tokenizer |
    chain-est-phone-lm \
        --no-prune-ngram-order=$no_prune_ngram_order \
        --ngram-order=$ngram_order \
        ark:- token_lm.fst.tmp

vocab_size=$(python -c "import cat; print(cat.shared.tokenizer.load('$tokenizer').vocab_size)")
python utils/tool/build_ctc_topo.py $vocab_size |
    fstcompile | fstarcsort --sort_type=olabel >T.fst.tmp || exit 1

fstcompose T.fst.tmp token_lm.fst.tmp |
    fstdeterminizestar --use-log=true >$w_specifier

rm {T,token_lm}.fst.tmp
echo "$0 done."
exit 0
