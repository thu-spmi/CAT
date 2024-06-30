'''
Copyright 2024 Tsinghua SPMI Lab, Author: Saier (sar_dar@163.com)
Sample sentence for Multilingual Subword generation. implements of Equation 2 in the paper https://arxiv.org/abs/2406.02166 
'''

import numpy as np
from collections import Counter
import random
# random.seed(216)
# np.random.seed(216)

a = 0.5                # a factor in the Eq.
target_size = 2959565   # num of target sentences
lang_list = ["en", "es", "fr", "it", "ky", "nl", "ru", "sv", "tr", "tt"]
source_path = "/mnt/workspace/liziwei/data/{}/excluded_train/text"
target_file = "/mnt/workspace/liziwei/data/ten/text_sampled"

def get_sentence(source_path):
    s_list = []    # [num of lang, num of sentence per lang]
    n_line = []    # num of sentence per lang
    for lang in lang_list:
        temp = []
        f_path = source_path.format(lang)
        f = open(f_path, 'r', encoding='utf-8')
        for line in f:
            temp.append(line.strip())
        f.close()
        n_line.append(len(temp))
        s_list.append(temp)
    return s_list, n_line

def get_q(n_line, a):
    n_sum = sum(n_line)
    p = []
    for item in n_line:
        p.append(item / n_sum)
    q = []
    p_sum = sum(np.asarray(p) ** a)
    for item in p:
        q.append(item**a/p_sum)
    return q

s_list, n_line = get_sentence(source_path)
q = get_q(n_line, a)
x = np.random.multinomial(n=1, pvals=q, size=target_size)   # 多项分布随机采样
print(f"Sample size: {Counter(np.argmax(x, axis=1).flatten())}")
with open(target_file, 'w', encoding='utf-8') as f:
    for i in np.argmax(x, axis=1):
        choice = random.choice(s_list[i])
        f.write(choice + "\n")
