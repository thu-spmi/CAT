"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Hongyu Xiang, Keyu An, Zheng Huahuan
"""

import kaldiio
import numpy as np
import argparse
import pickle
import h5py
from tqdm import tqdm


def ctc_len(label):
    extra = 0
    for i in range(len(label)-1):
        if label[i] == label[i+1]:
            extra += 1
    return len(label) + extra


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert to pickle")
    parser.add_argument("-f", "--format", type=str,
                        choices=["hdf5", "pickle"], default="pickle")
    parser.add_argument("-W", "--warning", action="store_true",
                        default=False)
    parser.add_argument("--describe", type=str, default=None,
                        help="Arithmetic expression used to describe sequence transformation by length. e.g. '(L+1)//3'")
    parser.add_argument("--filer", type=int, default=-1,
                        help="Allowing maximum length of sequences, otherwise which will be filtered out.")
    parser.add_argument("scp", type=str)
    parser.add_argument("label", type=str)
    parser.add_argument("weight", type=str)
    parser.add_argument("output_path", type=str)

    args = parser.parse_args()

    if args.warning:
        print(
            "Calculation of CTC loss requires the input sequence to be longer than ctc_len(labels)\n",
            "Check that in 'convert_to.py' if your model does subsampling on seq\n",
            "Make your modify at line 'if feature.shape[0] < ctc_len(label):' to filter unqualified seq\n",
            "If you have already done, ignore this.")

    label_dict = {}
    with open(args.label, 'r') as fi:
        lines = fi.readlines()
        for line in lines:
            sp = line.split()
            label_dict[sp[0]] = np.asarray([int(x) for x in sp[1:]])

    weight_dict = {}
    with open(args.weight, 'r') as fi:
        lines = fi.readlines()
        for line in lines:
            sp = line.split()
            weight_dict[sp[0]] = np.asarray([float(sp[1])])

    if args.format == "hdf5":
        h5_file = h5py.File(args.output_path, 'w')
    else:
        pickle_dataset = []

    count = 0
    L_MAX = args.filer if args.filer > 0 else float('inf')
    num_lines = sum(1 for line in open(args.scp, 'r'))
    if args.describe is None:
        def formated_L(x): return x
    else:
        # type: Callable[[int], int]
        formated_L = eval(f'lambda L: {args.describe}')

    f_opened = {}
    with open(args.scp, 'r') as fi:
        for line in tqdm(fi, total=num_lines):
            key, loc_ark = line.split()

            label = label_dict[key]
            weight = weight_dict[key]
            feature = kaldiio.load_mat(loc_ark, fd_dict=f_opened)

            if formated_L(feature.shape[0]) < ctc_len(label) or feature.shape[0] > L_MAX:
                count += 1
                continue

            if args.format == "hdf5":
                dset = h5_file.create_dataset(key, data=feature)
                dset.attrs['label'] = label
                dset.attrs['weight'] = weight
            else:
                pickle_dataset.append([key, loc_ark, label, weight])

    for f in f_opened.values():
        f.close()
    print(f"Remove {count} unqualified sequences in total.")
    if args.format == "pickle":
        with open(args.output_path, 'wb') as fo:
            pickle.dump(pickle_dataset, fo)
