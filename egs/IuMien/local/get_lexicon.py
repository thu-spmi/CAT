# Copyright 2023 Tsinghua SPMI Lab, Author: DongLukuan (330293721@qq.com)


spellphone_list = ['b','c','d','f','g','h','hl','hm','hn','hng','hny','j','k','l','m','mb','n','nd','ng','nj','nq','ny','nz','p','q','s','t','w','y','z','a','aa','aai','aau','ae','ai','au','e','ei','er','eu','i','o','oi','or','ou','u','v','h','','c','x','z']
spellphone_map = {'b':'p','c':'t͡sʰ','d':'t','f':'f','g':'k','h':'h','hl':'l̥','hm':'m̥','hn':'n̥','hng':'ŋ̊','hny':'ɲ̥','j':'t͡ɕ','k':'kʰ','l':'l','m':'m','mb':'b','n':'n','nd':'d','ng':'ŋ','nj':'d͡ʑ','nq':'ɡ','ny':'ɲ','nz':'d͡z','p':'pʰ','s':'s','t':'tʰ','w':'w','y':'j','z':'t͡s','a':'ɐ','aa':'a','aai':'aː ɪ','aau':'aː ʊ','ae':'æ','ai':'a ɪ','au':'a ʊ','e':'e','ei':'ɛ ɪ','er':'ɜ','eu':'ɜ o','i':'i','o':'o','oi':'oɪ','or':'ɒ','ou':'əu','u':'u'}
# spellphone_map = {'b':'p','c':'t͡sʰ','d':'t','f':'f','g':'k','h':'h','hl':'l̥','hm':'m̥','hn':'n̥','hng':'ŋ̊','hny':'ɲ̥','j':'t͡ɕ','k':'kʰ','l':'l','m':'m','mb':'b','n':'n','nd':'d','ng':'ŋ','nj':'d͡ʑ','nq':'ɡ','ny':'ɲ','nz':'d͡z','p':'pʰ','s':'s','t':'tʰ','w':'w','y':'j','z':'t͡s','a':'ɐ','aa':'a','aai':'aː ɪ','aau':'aː ʊ','ae':'æ','ai':'a ɪ','au':'a ʊ','e':'e','ei':'ɛ ɪ','er':'ɜ','eu':'ɜ o','i':'i','o':'o','oi':'o ɪ','or':'ɒ','ou':'ə u','u':'u'}

# spellphone_map = {'b':'p','c':'t͡s','d':'t','f':'f','g':'k','h':'h','hl':'l','hm':'m','hn':'n','hng':'ŋ','hny':'ɲ','j':'t͡ɕ','k':'k','l':'l','m':'m','mb':'b','n':'n','nd':'d','ng':'ŋ','nj':'d͡ʑ','nq':'ɡ','ny':'ɲ','nz':'d͡z','p':'p','s':'s','t':'t','w':'w','y':'j','z':'t͡s','a':'ɐ','aa':'a','aai':'a ɪ','aau':'a ʊ','ae':'æ','ai':'a ɪ','au':'a ʊ','e':'e','ei':'ɛ ɪ','er':'ɜ','eu':'ɜ o','i':'i','o':'o','oi':'o ɪ','or':'ɒ','ou':'ə u','u':'u'}

tones_map = {'h':'2','z':'4','x':'5'}

spellphone_list = sorted(spellphone_list,key=lambda x:len(x),reverse=True)

def phone_map(token_list):
    phone_list = []
    for i,token in enumerate(token_list):
        if token == 'q':
            if i == 0:
                phone_list.append('t͡ɕʰ')
                # phone_list.append('t͡ɕ')
            else:
                phone_list.append('ʔ')
        elif token == 'v':
            if token_list[i-1] in ['p','t','k','q']:
                phone_list.append('7')
            else:
                phone_list.append('3')
        elif token == 'c':
            if token_list[i-1] in ['p','t','k','q']:
                phone_list.append('8')
            else:
                phone_list.append('6')
        else:
            if i == len(token_list) - 1 and token in tones_map:
                phone_list.append(tones_map[token])
            else:
                phone_list.append(spellphone_map[token])
    return phone_list

def process_word2(word):
    # 基于元音辅音音调拆分单词
    start_index = 0
    word_compose = []
    while(start_index < len(word)):
        for token in spellphone_list:
            if start_index+len(token) <= len(word) and word[start_index:start_index+len(token)] == token:
                word_compose.append(token)
                start_index = start_index + len(token)
                break
    word_compose = phone_map(word_compose)
    return word_compose

import re

def merge_spaces(text):
    # 使用正则表达式将多个空格替换为一个空格
    return re.sub(r'\s+', ' ', text)

word_list = []
with open('./dict/word_list','r',encoding='utf-8') as f:
    for word in f:
        word_list.append(word.strip())

word_compose_list = []
for word in word_list:
    word_compose = process_word2(word)
    word_compose = merge_spaces(' '.join(word_compose))
    word_compose_list.append(word + '\t'+word_compose)
with open('./dict/lexicon.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(word_compose_list))