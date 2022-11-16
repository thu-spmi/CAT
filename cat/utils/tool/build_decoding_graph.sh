#!/bin/bash
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# Construct decoding graph for lexicon-based model.
set -e
<<"PARSER"
("tokenizerT", type=str, help="Path to the tokenizer for compiling T.fst")
("tokenizerG", type=str,
    help="Path to the tokenizer for compiling G.fst")
("lm_path", type=str, help="LM file (in ARPA format). This should be a token-based n-gram.")
("out_dir", type=str, help="Output directory. Decoding graph would be saved as <out_dir>/TLG.fst")
("-c", "--clean-auxiliary", action="store_true", help="Clean all temp files except the TLG.fst")
PARSER
eval $(python utils/parseopt.py $0 $*)

mkdir -p $out_dir
out_dir=$(readlink -f $out_dir)
lm_path=$(readlink -f $lm_path)

[ $tokenizerG == "-" ] &&
    tokenizerG=$tokenizerT

# lexicon.txt:
#     two t uː
#     to  t uː

# units.txt, this is slightly different from v2, which excludes the <blk> but add it later
#     <blk>  1
#     t  2
#     uː 3

# words.txt
#     <eps> 0
#     to    1
#     two   2
#     #0    3
#     <s>   4
#     </s>  5

# 1. add dummy prob ->
#    lexiconp.txt
#        two 1.0 t uː
#        to  1.0 t uː
# This command would generate:
#     units.txt lexiconp.txt words.txt in $out_dir
skip=1
for f in units.txt lexiconp.txt words.txt; do
    [ ! -f $out_dir/$f ] &&
        skip=0
done
[ $skip -eq 0 ] &&
    python utils/tool/prep_decoding_graph_materials.py \
        $tokenizerT $tokenizerG $out_dir

# 2. add disambiguous symbols ->
#    lexiconp_disambig.txt
#        two 1.0 t uː #1
#        to  1.0 t uː #2
# 3. build tokens.txt: <eps> 0 + units.txt + #0 + disambiguous list
#    tokens.txt
#        <eps> 0
#        <blk> 1
#        t  2
#        uː 3
#        #0 4
#        #1 5
#        #2 6
[ -z $KALDI_ROOT ] && {
    echo "\$KALDI_ROOT is not set. re-run with"
    echo "KALDI_ROOT=xxx $0 $*"
    exit 1
}
export KALDI_ROOT=$KALDI_ROOT
! [[ -d $KALDI_ROOT && -d $KALDI_ROOT/egs/wsj/s5 ]] && (
    echo "kaldi tool at '$KALDI_ROOT' not installed."
    exit 1
)
cd $KALDI_ROOT/egs/wsj/s5 && . ./path.sh

n_units=$(tail -n 1 $out_dir/units.txt | awk '{print $2}')
ndisambig=$(utils/add_lex_disambig.pl $out_dir/lexiconp.txt $out_dir/lexiconp_disambig.txt)
(for n in $(seq 0 $ndisambig); do echo "#$n $(($n + 1 + $n_units))"; done) >$out_dir/disambig.txt
echo "<eps> 0" | cat - $out_dir/units.txt $out_dir/disambig.txt >$out_dir/tokens.txt

# 4. determine index of #0 in tokens.txt and words.txt
#    e.g. tokens.txt: #0 -> 4; words.txt: #0 -> 3
# 5. construct L_disambig.fst
utils/make_lexicon_fst.pl --pron-probs $out_dir/lexiconp_disambig.txt |
    fstcompile --isymbols=$out_dir/tokens.txt --osymbols=$out_dir/words.txt \
        --keep_isymbols=false --keep_osymbols=false |
    fstaddselfloops "echo $(grep \#0 $out_dir/tokens.txt | awk '{print $2}') |" \
        "echo $(grep \#0 $out_dir/words.txt | awk '{print $2}') |" |
    fstarcsort --sort_type=olabel >$out_dir/L_disambig.fst

# 6. construct G.fst from lm arpa file
[ ! -f $out_dir/G.fst ] &&
    arpa2fst --disambig-symbol=#0 \
        --max-arpa-warnings=5 \
        --read-symbol-table=$out_dir/words.txt \
        $lm_path $out_dir/G.fst

# 6.1 check G, the first number should be small
# ... if it's not, I guess there're many words in your lm.arpa 
# ... which are not present in --read-symbol-table
fstisstochastic $out_dir/G.fst || true

# 7. Compose TLG.fst
cd - >/dev/null
[ ! -f $out_dir/T_disambig.fst ] &&
    python utils/tool/build_ctc_topo.py --build-for-decode \
        $n_units --n-disambig $((1 + $ndisambig)) |
    fstcompile >$out_dir/T_disambig.fst

[ ! -f $out_dir/TLG.fst ] &&
    fsttablecompose $out_dir/L_disambig.fst $out_dir/G.fst |
    fstdeterminizestar |
        fstminimizeencoded |
        fstarcsort --sort_type=ilabel |
        fsttablecompose $out_dir/T_disambig.fst - \
            >$out_dir/TLG.fst

[ $clean_auxiliary == "True" ] && {
    rm -f $out_dir/{G,L_disambig,T_disambig}.fst
    rm -f $out_dir/{disambig,lexiconp_disambig,lexiconp,tokens,units,words}.txt
}
echo "$0 done."
exit 0
