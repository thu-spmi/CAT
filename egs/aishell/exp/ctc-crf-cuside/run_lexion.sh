#!/bin/bash

dir=$(dirname $0)
mkdir -p $dir/prepare_lexicon

cd $dir/prepare_lexicon
[[ ! -f lexicon.txt || ! -f dict.txt ]] && {
    [ ! -f resource_aishell.tgz ] &&
        wget https://www.openslr.org/resources/33/resource_aishell.tgz

    [ ! -f lexicon.txt ] && {
        tar -zxf resource_aishell.tgz
        mv resource_aishell/lexicon.txt ./
    }

    [ ! -f dict.txt ] && (
        # prepare word segmentation dictionary for jieba token
        cut <lexicon.txt -f 1 |
            awk {'print $1, 99'} >dict.txt
    )
}
echo "finsh lexicon and dict"
