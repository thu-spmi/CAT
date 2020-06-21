import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import sys

class SpeechDataset(Dataset):
    def __init__(self, h5py_path):
        self.h5py_path = h5py_path
        self.hdf5_file = h5py.File(h5py_path, 'r')
        self.keys = list(self.hdf5_file.keys())

    def __del__(self):
        self.hdf5_file.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        dataset = self.hdf5_file[self.keys[idx]]
        mat = dataset[()]
        label = dataset.attrs['label']
        weight = dataset.attrs['weight']
        return torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)

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
          self.data_batch.append([torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)])

        hdf5_file.close()
        print("read all data into memory")

    def __len__(self):
        return len(self.data_batch)

    def __getitem__(self, idx):
        return self.data_batch[idx]


def pad_tensor(t, pad_to_length, dim):
    pad_size = list(t.shape)
    pad_size[dim] = pad_to_length - t.size(dim)
    return torch.cat([t, torch.zeros(*pad_size).type_as(t)], dim=dim)

class PadCollate:
    def __init__(self):
        pass
    def __call__(self, batch):
        # batch: list of (mat, label, weight)
        # return: logits, input_lengths, label_padded, label_lengths, weights
        input_lengths = map(lambda x: x[0].size(0), batch)
        if sys.version > '3':
            input_lengths = list(input_lengths)
        max_input_length = max(input_lengths)
        label_lengths = map(lambda x: x[1].size(0), batch)
        if sys.version > '3':
            label_lengths = list(label_lengths)
        max_label_length = max(label_lengths)
        input_batch = map(lambda x:pad_tensor(x[0], max_input_length, 0), batch)
        label_batch = map(lambda x:pad_tensor(x[1], max_label_length, 0), batch)
        if sys.version > '3':
            input_batch = list(input_batch)
            label_batch = list(label_batch) 
        logits = torch.stack(input_batch, dim=0)
        label_padded = torch.stack(label_batch, dim=0)
        input_lengths = torch.IntTensor(input_lengths)
        label_lengths = torch.IntTensor(label_lengths)
        weights = torch.FloatTensor([x[2] for x in batch])
        return logits, input_lengths, label_padded, label_lengths, weights
