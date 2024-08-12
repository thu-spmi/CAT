# Copyright 2023 Tsinghua SPMI Lab, Author: DongLukuan (330293721@qq.com)


train_data_path = './MightLJSpeech-1.1/train_data.txt'
dev_data_path = './MightLJSpeech-1.1/dev_data.txt'
test_data_path = './MightLJSpeech-1.1/test_data.txt'

def read_file(file_path,need_split=False):
    file_text = []
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f:
            if need_split:
                file_text.append(line.strip().split('\t')[1])
            else:
                file_text.append(line.strip())
    return file_text

train_data = read_file(train_data_path,True)
dev_data = read_file(dev_data_path)
test_data = read_file(test_data_path)

new_test_data = []
count = 0
for line in test_data:
    ids,sentence = line.strip().split('\t',maxsplit=1)
    # print(sentence)
    if sentence in train_data:
        count = count + 1
        continue
    else:
        new_test_data.append(line)

with open(test_data_path,'w',encoding='utf-8') as f:
    write_text = '\n'.join(new_test_data)
    f.write(write_text)

new_dev_data = []
count = 0
for line in dev_data:
    ids,sentence = line.strip().split('\t',maxsplit=1)
    # print(sentence)
    if sentence in train_data:
        count = count + 1
        continue
    else:
        new_dev_data.append(line)

with open(dev_data_path,'w',encoding='utf-8') as f:
    write_text = '\n'.join(new_dev_data)
    f.write(write_text)