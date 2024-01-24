import json
import numpy as np
import math
import scipy.stats as st
import argparse
def compute_P(N10, K):
    w=(abs(N10-K/2)-0.5)/math.sqrt(K/4)
    P=2*(1-st.norm(0,1).cdf(abs(w)))
    return P

def McNemar(list1, list2):
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
