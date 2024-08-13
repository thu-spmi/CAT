# Copyright 2023 Tsinghua SPMI Lab, Author: DongLukuan (330293721@qq.com)

text_path = './data/src/train/text'

word_set = set()
with open(text_path,'r',encoding='utf-8') as f:
    for line in f:
        try:
            ids,sentence = line.strip().split('\t')
        except:
            print(line.strip().split('\t'))
        word_set.update(sentence.split(' '))
        # word_set(set(sentence.split(' ')))
word_set = sorted(word_set)

word_list_path = './dict/word_list'
with open(word_list_path,'w',encoding='utf-8') as f:
    for word in word_set:
        f.write(word+'\n')