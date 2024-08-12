# Copyright 2023 Tsinghua SPMI Lab, Author: DongLukuan (330293721@qq.com)

import random
import argparse

def main_work(args):
    # 打开a.txt文件进行读取
    data_path=args.data_path
    if args.data_output_path is None:
        output_path = data_path
    else:
        output_path = args.data_output_path
    with open(data_path, 'r') as input_file:
        # 读取a.txt的内容
        data = input_file.readlines()

    # 将所有的|替换为\t
    data = [line.replace('|', '\t').replace('.wav','') for line in data]

    # 随机打乱数据
    random.shuffle(data)

    # 将数据划分为10份
    num_parts = 10
    chunk_size = len(data) // num_parts
    chunks = [data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_parts)]

    # 处理剩余的数据（如果有的话）
    remainder = len(data) % num_parts
    for i in range(remainder):
        chunks[i].append(data[num_parts * chunk_size + i])

    select_train_data = [0,1,2,3,4,5,6,7]

    with open(f'{output_path}/train_data_test.txt', 'w', encoding='utf-8') as file:
        for i in select_train_data:
            file.writelines(chunks[i])
    with open(f'{output_path}/dev_data_test.txt', 'w', encoding='utf-8') as file:
        file.writelines(chunks[8])
    with open(f'{output_path}/test_data_test.txt', 'w', encoding='utf-8') as file:
        file.writelines(chunks[9])


if __name__ == "__main__":
        # 创建解析器对象
    parser = argparse.ArgumentParser(description="split data")
    
    # 添加位置参数
    parser.add_argument("--data_path", type=str, help="MightLJSpeech data path")
    parser.add_argument("--data_output_path", type=str, help="splt data output path",default=None)
    # 解析参数
    args = parser.parse_args()
    main_work(args)
    
