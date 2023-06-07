#!/bin/bash
#
# copyright 2023 Tsinghua University
# author: Huahuan Zheng
#
# Prepare IPA-based lexicon
set -e -u

<<"PARSER"
("-lang", type=str, help="Language annotation, e.g., en, zh, ...")
("-text", type=str, default="data/src/${lang}-excluded_train/text",
    help="Texts for collecting graphemes, usually the transcript corpus. "
    "If not specified, 'data/src/${lang}-excluded_train/text' would be used.")
("-char", action="store_true", help="Identify the language as character-based (defaultly word-based).")
("-g2p_model", type=str,
    help="Path of the G2P model. You can download models from http://www.isle.illinois.edu/speech_web_lg/data/g2ps/")
("-dst", default="data/lang-${lang}", help="Ouput dicrectory, default: data/lang-${lang}")
PARSER
eval $(python utils/parseopt.py $0 $*)

[[ -z $g2p_model || ! -f $g2p_model ]] && {
    echo "G2P model: '$g2p_model' not found." >&2
    exit 1
}
export PATH="../../src/bin:$PATH"
[ ! $(command -v phonetisaurus-apply) ] && {
    echo "command 'phonetisaurus-apply' not found in the \$PATH" >&2
    echo "... check if it exists in ../../src/bin/" >&2
    exit 1
}

if [ $dst == "None" ]; then
    dlang="data/lang-$lang"
else
    dlang="$dst"
fi
unset dst
mkdir -p $dlang
[ ! -z "$(ls -A $dlang)" ] && {
    echo "warning: destination folder '$dlang' is not empty," >&2
    echo "... which may cause unexpected mistakes." >&2
    echo "... Please consider clean it then re-run the script." >&2
}

export LC_ALL=C.UTF-8

[ ! -f $dlang/wordlist ] && {
    cmd_wlist="<$text cut -f 2- |"
    if [ $char == "True" ]; then
        cmd_wlist="$cmd_wlist grep -o . |"
    else
        cmd_wlist="$cmd_wlist tr ' ' '\n' |"
    fi
    cmd_wlist="$cmd_wlist sort -u -s >$dlang/wordlist"
    eval "$cmd_wlist"
}

[ ! -f $dlang/lexicon ] &&
    phonetisaurus-apply --model $g2p_model \
        --word_list $dlang/wordlist \
        >$dlang/lexicon

# post-process the lexicon to replace missing non-IPA symbols
## fetch the ipa list
phonetic_data="local/data"
mkdir -p $phonetic_data
[ ! -f $phonetic_data/ipa_all.csv ] &&
    wget https://raw.githubusercontent.com/dmort27/panphon/master/panphon/data/ipa_all.csv \
        -q -O $phonetic_data/ipa_all.csv
[ ! -f $phonetic_data/ipa.list ] &&
    tail -n +2 $phonetic_data/ipa_all.csv | cut -d ',' -f 1 | sort -u -s >$phonetic_data/ipa.list

[ ! -f $dlang/lexicon.bak ] && mv $dlang/lexicon{,.bak}
python local/repl_nonIPA.py \
    $dlang/lexicon.bak $phonetic_data/ipa.list \
    --extend $phonetic_data/ipa_extend.txt \
    >$dlang/lexicon || exit 1

# prepare phone units
cut <$dlang/lexicon -f 2- |
    tr ' ' '\n' | sort -u -s >$dlang/phonemes.txt

# get the ipa mapping matrix (for join-AP training only)
python local/get_ipa_mapping.py \
    $dlang/phonemes.txt \
    $phonetic_data/ipa_all.csv \
    $dlang/${lang}-pv.npy || exit 1

echo "$0 done" && exit 0
