"""
Copyright 2021 Tsinghua University
Apache 2.0.

Author: Chengrui Zhu  2021
        Wenjie Peng   2021

This script implements multi/cross-lingual related functions originally written by Chengrui Zhu,
which is latter refactored by Wenjie Peng.
"""

import json
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn

def get_idx_by_key(dic, key):
    idx = 0
    for k in dic.keys():
        if k == key:
            return idx
        idx += 1
    exit(1)

def load_token_idx(fin):
    idx = OrderedDict()
    with open(fin, "r") as fp:
        for line in fp.readlines()[1:]:
            line = line.strip().split()
            k, v = line[0], int(line[1])-1
            if k == "#0":
                break
            idx.update({k:v})
    return idx


def load_src_des_idx(f1, f2):

    idx1 = load_token_idx(f1)
    idx2 = load_token_idx(f2)
    src_idx = []
    des_idx = []
    for i in range(len(idx2)):
        k = list(idx2.keys())[i]
        if k in idx1:
            v = idx1[k]
            src_idx.append(v)
            des_idx.append(i)

    return src_idx, des_idx


def load_des_idx(fin):
    idx = load_token_idx(fin)
    return list(idx.values())


def load_pv(fin):

    pv = np.load(fin)
    pv = torch.Tensor(pv)
    return pv


def load_mc_conf(args):

    conf = args.mc_conf
    with open(conf, "r") as fp:
        config = json.load(fp)
    src_idx, des_idx = load_src_des_idx(config["src_token"], config["des_token"])
#    des_idx = load_des_idx(config["des_token"])
    
    pv = load_pv(config["P"])
    hdim = config["hdim"]
    odim = config["odim"]
    mode = config["mode"]
    usg = config["usg"]
    lr = config["lr"]

    return src_idx, des_idx, pv, hdim, odim, mode, usg, lr

def update_model(model, ckpt, args, loc):

    src_idx, des_idx, pv, hdim, odim, mode, usg, lr = load_mc_conf(args)
    model = reset_model(model, ckpt, loc, mode, usg, pv, src_idx, des_idx, hdim, odim)
    return model, lr


def reset_model(model, ckpt, loc, mode, usg, pv, src_idx, des_idx, hdim, odim):
    """
    Reset neural network topology for multilingual finetune..
    Args   :
        ckpt    : saved checkpoint of trained model.
        loc     : device location.
        mode    : [flat_phone|join_ap]
        usg     : [finetune|eval]
        pv      : phonological vector matrix
        src_idx : index of the seen AM units in the last hidden layer of the pretrained model for finetune
        des_idx : index of the seen AM units in the last hidden layer of the new model for finetune
    Returns:
    """

    # new transformation matrix before Softmax layer
    assert mode in ["flat_phone", "joinap_linear", "joinap_nonlinear"]
    assert usg in ["finetune", "zero-shot-eval", "few-shot-eval", "multi-eval", "multi-finetune-eval"]

    if usg == "few-shot-eval" or usg == "multi-finetune-eval":
        return model.to(device=loc)
    else:
        if mode == "flat_phone":
            assert usg in ["finetune", "multi-eval"]
            new_linear = nn.Linear(hdim, odim)

            new_linear.weight.requires_grad = False
            new_linear.bias.requires_grad = False
            new_linear.to(device=loc)
            if "eval" in usg:
                new_linear.weight[des_idx] = ckpt["linear.weight"][src_idx]
                new_linear.bias[des_idx] = ckpt["linear.bias"][src_idx]
            else:
                new_linear.weight[des_idx] = ckpt["model"]["module.infer.linear.weight"][src_idx]
                new_linear.bias[des_idx] = ckpt["model"]["module.infer.linear.bias"][src_idx]
            new_linear.weight.requires_grad = True
            new_linear.bias.requires_grad = True

            if "eval" in usg:
                ckpt["linear.weight"] = new_linear.weight
                ckpt["linear.bias"] = new_linear.bias
                model.linear = new_linear
                model.load_state_dict(ckpt)
            else:
                ckpt["model"]["module.infer.linear.weight"] = new_linear.weight
                ckpt["model"]["module.infer.linear.bias"] = new_linear.bias
                ckpt["scheduler"]["best_metric"] = None
                model.module.infer.linear = new_linear
                model.load_state_dict(ckpt["model"])

        else:
            assert usg in ["finetune", "zero-shot-eval"]
            P = nn.Parameter(pv, requires_grad = False)
            p_c = P.size()[0]
            new_linear = nn.Linear(p_c, odim)

            if mode == "joinap_linear":
                if "eval" in usg:
                    model.P = P
                    ckpt["linear.weight"] = new_linear.weight
                    ckpt["linear.bias"] = new_linear.bias
                    ckpt["P"] = P
                    model.linear = new_linear
                    model.load_state_dict(ckpt)
                else:
                    model.module.infer.P = P
                    ckpt["model"]["module.infer.P"] = P
                    ckpt["model"]["module.infer.linear.weight"] = new_linear.weight
                    ckpt["model"]["module.infer.linear.bias"] = new_linear.bias
                    ckpt["scheduler"]["best_metric"] = None
                    model.module.infer.linear = new_linear
                    model.load_state_dict(ckpt["model"])
            else:
                if "eval" in usg:
                    model.P = P
                    ckpt["P"] = P
                    ckpt["linear.weight"] = new_linear.weight
                    ckpt["linear.bias"] = new_linear.bias
                    model.linear = new_linear
                    model.load_state_dict(ckpt)
                else:
                    model.module.infer.P = P
                    ckpt["model"]["module.infer.P"] = P
                    ckpt["model"]["module.infer.linear.weight"] = new_linear.weight
                    ckpt["model"]["module.infer.linear.bias"] = new_linear.bias
                    ckpt["scheduler"]["best_metric"] = None
                    model.module.infer.linear = new_linear
                    model.load_state_dict(ckpt["model"])

    return model.to(device=loc)

