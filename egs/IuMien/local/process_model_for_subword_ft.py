# Copyright 2023 Tsinghua SPMI Lab, Author: DongLukuan (330293721@qq.com)

import torch
import torch.nn as nn 
import argparse


def main_work(args):
    input_model_path = args.pt_model_path
    output_model_path = args.output_model_path
    vocab_size = args.vocab_size
    ckpt = torch.load(input_model_path)
    model = ckpt['model']
    new_linear = nn.Linear(model["module.encoder.classifier.weight"].shape[1], vocab_size)
    model["module.encoder.classifier.weight"] = new_linear.weight
    model["module.encoder.classifier.bias"] = new_linear.bias

    ckpt['model'] = model 
    torch.save(ckpt, output_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="split data")
    
    # 添加位置参数
    parser.add_argument("--pt_model_path", type=str, help="pretrain model path")
    parser.add_argument("--output_model_path", type=str, help="model output path")
    parser.add_argument("--vocab_size", type=int, help="vocab size")

    # 解析参数
    args = parser.parse_args()
    main_work(args)