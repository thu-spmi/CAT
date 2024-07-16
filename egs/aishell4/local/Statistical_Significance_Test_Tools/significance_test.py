# Copyright 2020 Tsinghua SPMI Lab 
# Apache 2.0.
# Author: Xiangzhu Kong(kongxiangzhu99@gmail.com), Hong Liu
#
# Description:
#   This script contains functions to perform statistical hypothesis testing using McNemar's test and the matched-pair t-test. 
#   The key functions include computing the p-value for McNemar's test, computing the p-value for the matched-pair t-test, and 
#   processing input data from JSON files to perform these tests.

import json
import numpy as np
import math
import scipy.stats as st
import argparse

def compute_P(N10, K):
    """
    Compute the p-value for McNemar's test.

    Args:
        N10 (int): Number of samples where the first condition is true and the second condition is false.
        K (int): Total number of discordant pairs.

    Returns:
        float: p-value for McNemar's test.
    """
    w=(abs(N10-K/2)-0.5)/math.sqrt(K/4)
    P=2*(1-st.norm(0,1).cdf(abs(w)))
    return P

def McNemar(list1, list2):
    """
    Perform McNemar's test on two lists of binary outcomes.

    Args:
        list1 (list): First list of binary outcomes.
        list2 (list): Second list of binary outcomes.

    Returns:
        float: p-value for McNemar's test.
    """
    c1=0
    c2=0
    for t1, t2 in zip(list1, list2):
        if t1 and not t2:
            c1+=1
        elif not t1 and t2:
            c2+=1
    P=compute_P(c1, c1+c2)
    return P

def matched_pair(list1, list2):
    """
    Perform a matched-pair t-test on two lists of continuous outcomes.

    Args:
        list1 (list): First list of continuous outcomes.
        list2 (list): Second list of continuous outcomes.

    Returns:
        float: p-value for the matched-pair t-test.
    """
    n=len(list1)
    Z=[item1-item2 for item1, item2 in zip(list1, list2)]
    u=np.mean(Z)
    sigma=math.sqrt(sum([(z-u)**2 for z in Z])/(n-1))
    w=u*math.sqrt(n)/sigma
    P=2*(1-st.norm(0,1).cdf(abs(w)))
    return P
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path1", type=str, help="The result file path (.json) of exp1")
    parser.add_argument("result_path2", type=str, help="The result file path (.json) of exp2")
    parser.add_argument("--method", type=str, default='mp', help="Matched pair test (mp) or McNemar test (mc)")
    args = parser.parse_args()
    list1 = json.load(open(args.result_path1, 'r'))
    list2 = json.load(open(args.result_path2, 'r'))
    test_func = McNemar if args.method=='mc' else matched_pair 
    print('p:', test_func(list1, list2))
