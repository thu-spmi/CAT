import h5py
import kaldi_io
import numpy as np
import argparse
import os
import math

def ctc_len(label):
    extra = 0 
    for i in range(len(label)-1):
        if label[i] == label[i+1]:
            extra += 1
    return len(label) + extra


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert to hdf5")
    parser.add_argument("scp", type=str)
    parser.add_argument("label", type=str)
    parser.add_argument("weight", type=str)
    parser.add_argument("chunk_size", type=int, default=40)
    parser.add_argument("hdf5", type=str)
    args = parser.parse_args()

    label_dict = {}
    with open(args.label) as f:
        lines = f.readlines()
        for line in lines:
            sp = line.split()
            label_dict[sp[0]] = np.asarray([int(x) for x in sp[1:]])

    weight_dict = {}
    with open(args.weight) as f:
        lines = f.readlines()
        for line in lines:
            sp = line.split()
            weight_dict[sp[0]] = np.asarray([float(sp[1])])

    with open(args.scp) as f:
        lines = f.readlines()
        for line in lines:
            key, value = line.split()

            label = label_dict[key]
            weight = weight_dict[key]
            feature = kaldi_io.read_mat(value)
            feature = np.asarray(feature)
	  
            if feature.shape[0] < ctc_len(label):
                print('{} is too short'.format(key))
                continue
            cate = str(math.ceil(feature.shape[0]/args.chunk_size))+'.hdf5'
            current_dir = os.path.join(args.hdf5,cate)
            h5_file = h5py.File(current_dir, 'a')
            dset = h5_file.create_dataset(key, data=feature)
            dset.attrs['label'] = label
            dset.attrs['weight'] = weight
            h5_file.close()
