import numpy as np
import torch
import sys
import kaldi_io
import pickle
from torch.utils.data import Dataset, DataLoader

class SpeechDataset(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path) as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        key, feature_path, label, weight = self.dataset[idx]
        mat = np.asarray(kaldi_io.read_mat(feature_path))
        return torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)


class SpeechDatasetMem(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path) as f:
            self.dataset = pickle.load(f)

        self.data_batch = []

        for data in self.dataset:
          key, feature_path, label, weight = data
          mat = np.asarray(kaldi_io.read_mat(feature_path))
          self.data_batch.append([torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)])
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
        max_input_length = max(input_lengths)
        label_lengths = map(lambda x: x[1].size(0), batch)
        max_label_length = max(label_lengths)

        input_batch = map(lambda x:pad_tensor(x[0], max_input_length, 0), batch)
        label_batch = map(lambda x:pad_tensor(x[1], max_label_length, 0), batch)
        logits = torch.stack(input_batch, dim=0)
        label_padded = torch.stack(label_batch, dim=0)
        input_lengths = torch.IntTensor(input_lengths)
        label_lengths = torch.IntTensor(label_lengths)
        weights = torch.FloatTensor([x[2] for x in batch])
        return logits, input_lengths, label_padded, label_lengths, weights
