# Author: Huahuan Zheng (maxwellzh@outlook.com)
# This script includes the processing of librispeech extra corpus text
set -e -u

d_out=data

mkdir -p $d_out
text=$d_out/librispeech.txt
if [ ! -f $text ]; then
    archive=$d_out/librispeech-lm-norm.txt.gz
    if [ ! -f $archive ]; then
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P $d_out || exit 1
    fi
    # check archive
    if [ $(md5sum $archive | cut -d ' ' -f 1) != "c83c64c726a1aedfe65f80aa311de402" ]; then
        echo "MD5 checking failed for $archive, please rm it then run this script again."
        exit 1
    fi
    gunzip -c $archive >$text || exit 1
    rm $archive
    echo "Fetched librispeech extra text corpus at $text"
else
    echo "$text file exist. skipped"
fi

# check md5sum
if [ $(md5sum $text | cut -d ' ' -f 1) != "c8288034566b62698db24f6cd414160d" ]; then
    echo "MD5 checking failed for $text, please rm it then run this script again."
    exit 1
fi
