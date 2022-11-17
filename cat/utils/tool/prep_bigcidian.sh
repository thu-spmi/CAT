# get bigcidian for segmentation or phone-based training
# author: Huahuan Zheng

set -e
set -u
<<"PARSER"
("-dir_dict", type=str, default="data/local/dict", 
    help="Directory to prepare dictionary file. default: data/local/dict")
PARSER
eval $(python utils/parseopt.py $0 $*)

mkdir -p $dir_dict
[ ! -f $dir_dict/lexicon.txt ] && (
    [ ! -f $dir_dict/run.sh ] && (
        # git clone https://github.com/speechio/BigCiDian.git $dir_dict
        git clone git@github.com:speechio/BigCiDian.git $dir_dict
    )
    cd $dir_dict/EN
    sh run.sh >>prep.log
    cd - >/dev/null

    cd $dir_dict/CN
    sh run.sh >>prep.log
    cd - >/dev/null

    cat $dir_dict/EN/EN.txt $dir_dict/CN/CN.txt | sort -u >$dir_dict/lexicon.txt
)

[ ! -f $dir_dict/dict.txt ] && (
    # prepare dictionary for jieba tokenizer
    # must be separated by space instead of tab
    cut <$dir_dict/lexicon.txt -f 1 |
        awk {'print $1, 99'} >$dir_dict/dict.txt
)

echo "prepared:"
echo "  lexicon:    $dir_dict/lexicon.txt"
echo "  dictionary: $dir_dict/dict.txt"
echo "$0 done."
exit 0
