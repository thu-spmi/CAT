"""
Copyright 2021 Tsinghua University
Apache 2.0.
Author: Hongyu Xiang, Keyu An, Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
"""

import os
import kaldiio
import h5py
import coreutils
import pickle
import math
from typing import Tuple, Sequence, List, Optional

import torch
from torch.utils.data import Dataset


class FeatureReader:
    def __init__(self) -> None:
        self._opened_fd = {}

    def __call__(self, arkname: str):
        return kaldiio.load_mat(arkname, fd_dict=self._opened_fd)

    def __del__(self):
        for f in self._opened_fd.values():
            f.close()
        del self._opened_fd


class SpeechDataset(Dataset):
    def __init__(self, h5py_path):
        self.h5py_path = h5py_path
        self.dataset = None
        hdf5_file = h5py.File(h5py_path, 'r')
        self.keys = list(hdf5_file.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5py_path, 'r')

        dataset = self.dataset[self.keys[idx]]
        mat = dataset[:]
        label = dataset.attrs['label']
        weight = dataset.attrs['weight']

        return torch.tensor(mat, dtype=torch.float), torch.IntTensor(label), torch.tensor(weight, dtype=torch.float)


class SpeechDatasetMem(Dataset):
    def __init__(self, h5py_path):
        hdf5_file = h5py.File(h5py_path, 'r')
        keys = hdf5_file.keys()
        self.data_batch = []
        for key in keys:
          dataset = hdf5_file[key]
          mat = dataset[()]
          label = dataset.attrs['label']
          weight = dataset.attrs['weight']
          self.data_batch.append(
              [torch.tensor(mat, dtype=torch.float), torch.IntTensor(label), torch.tensor(weight, dtype=torch.float)])

        hdf5_file.close()
        print("read all data into memory")

    def __len__(self):
        return len(self.data_batch)

    def __getitem__(self, idx):
        return self.data_batch[idx]


class SpeechDatasetPickle(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)
        self.freader = FeatureReader()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, feature_path, label, weight = self.dataset[idx]
        mat = self.freader(feature_path)
        return torch.tensor(mat, dtype=torch.float), torch.IntTensor(label), torch.tensor(weight, dtype=torch.float)


class SpeechDatasetMemPickle(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)

        self.data_batch = []
        freader = FeatureReader()

        for data in self.dataset:
            key, feature_path, label, weight = data
            mat = freader(feature_path)
            self.data_batch.append(
                [torch.tensor(mat, dtype=torch.float), torch.IntTensor(label), torch.tensor(weight, dtype=torch.float)])

    def __len__(self):
        return len(self.data_batch)

    def __getitem__(self, idx):
        return self.data_batch[idx]


class InferDataset(Dataset):
    def __init__(self, scp_path) -> None:
        super().__init__()
        with open(scp_path, 'r') as fi:
            lines = fi.readlines()
        self.dataset = [x.split() for x in lines]
        self.freader = FeatureReader()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        key, feature_path = self.dataset[index]
        mat = self.freader(feature_path)
        return key, torch.tensor(mat, dtype=torch.float), torch.LongTensor([mat.shape[0]])


class sortedPadCollate():
    def __call__(self, batch):
        """Collect data into batch by desending order and add padding.

        Args: 
            batch  : list of (mat, label, weight)
            mat    : torch.FloatTensor
            label  : torch.IntTensor
            weight : torch.FloatTensor

        Return: 
            (logits, input_lengths, labels, label_lengths, weights)
        """
        batches = [(mat, label, weight, mat.size(0))
                   for mat, label, weight in batch]
        batch_sorted = sorted(batches, key=lambda item: item[3], reverse=True)

        mats = coreutils.pad_list([x[0] for x in batch_sorted])

        labels = torch.cat([x[1] for x in batch_sorted])

        input_lengths = torch.LongTensor([x[3] for x in batch_sorted])

        label_lengths = torch.IntTensor([x[1].size(0) for x in batch_sorted])

        weights = torch.cat([x[2] for x in batch_sorted])

        return mats, input_lengths, labels, label_lengths, weights
