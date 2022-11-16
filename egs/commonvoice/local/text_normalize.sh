#!/bin/bash
# Author: Huahuan Zheng (maxwellzh@outlook.com)
# Text normalize
set -e -u

for file in $(python -c "import json;\
print(' '.join(x['trans'] for x in json.load(open('data/metainfo.json', 'r')).values()))"); do
    [ ! -f $file.bak ] && mv $file $file.bak
    cut <$file.bak -f 2- | sed -e 's/[.]//g; s/!//g; s/?//g' \
        -e 's/“//g; s/"//g; s/,//g; s/”//g' \
        -e "s/'//g; s/’//g; s/‘//g" \
        -e 's/://g; s/[;]//g; s/[(]//g; s/[)]//g;' \
        -e 's/[\]//g' |
        tr '[:upper:]' '[:lower:]' >$file.trans.tmp

    cut <$file.bak -f 1 >$file.id.tmp
    paste $file.{id,trans}.tmp >$file
    rm -rf $file.{id,trans}.tmp
done
