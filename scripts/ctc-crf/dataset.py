'''
Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
Apache 2.0.
(Class SpecAugmentor is finished by Kai Hu)
'''

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import math
import kaldi_io
import h5py
if sys.version > '3':
    import pickle
else:
    import cPickle as pickle


class Augmentor(object):
    def __init__(self):
        super(Augmentor, self).__init__()

    def __call__(self, input):
        raise NotImplementedError

# Google_interspeech2019 SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.
# SpecAugment consists of three kinds of deformations of the spectrogram (1.Time warping 2.Frequency masking 3.Time masking)
# Here, we adopt frequency masking and time masking.(Because the time warping's effect is small)
# in our experiments, 80fbank with delta, delta-delta and a 3-fold reduced frame rate is used.
# Dataset  freq_mask_param  num_freq_mask  time_mask_param  time_mask_ratio  num_time_mask
#   SWBD         15              2              23              0.2              2


class SpecAugmentor(Augmentor):
    def __init__(self, freq_mask_param, num_freq_mask, time_mask_param, time_mask_ratio, num_time_mask):
        super(SpecAugmentor, self).__init__()
        self._freqmp = freq_mask_param
        self._nfreqm = num_freq_mask
        self._timemp = time_mask_param
        self._timemr = time_mask_ratio
        self._ntimem = num_time_mask

    def __call__(self, tensor):
        batch_size, length, dim = tensor.size()
        num_freq_bins = int(dim/3)  # original delta delta-delta
        for n in range(batch_size):
            # frequency masking
            for i in range(self._nfreqm):
                f = np.random.randint(0, self._freqmp)
                if f == 0:
                    continue
                f_start = np.random.randint(0, num_freq_bins - f)
                tensor[n, :, f_start:f_start + f] = 0.0
                tensor[n, :, f_start + num_freq_bins:f_start +
                       f + num_freq_bins] = 0.0
                tensor[n, :, f_start + 2 * num_freq_bins:f_start +
                       f + 2 * num_freq_bins] = 0.0
            # time masking
            total_time_masked_length = 0
            for i in range(self._ntimem):
                t = np.random.randint(0, self._timemp)
                if total_time_masked_length + t > self._timemr * length:
                    break
                if t == 0:
                    continue
                t_start = np.random.randint(0, length - t)
                tensor[n, t_start:t_start + t, :] = 0.0
                total_time_masked_length += t
        return tensor


class SpeechDataset(Dataset):
    def __init__(self, h5py_path):
        self.h5py_path = h5py_path
        hdf5_file = h5py.File(h5py_path, 'r')
        self.keys = hdf5_file.keys()
        hdf5_file.close()
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        hdf5_file = h5py.File(self.h5py_path, 'r')
        dataset = hdf5_file[self.keys[idx]]
        mat = dataset.value
        label = dataset.attrs['label']
        weight = dataset.attrs['weight']
        hdf5_file.close()
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

class SpeechDatasetPickle(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        key, feature_path, label, weight = self.dataset[idx]
        mat = np.asarray(kaldi_io.read_mat(feature_path))
        return torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)


class SpeechDatasetMemPickle(Dataset):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)

        self.data_batch = []

        for data in self.dataset:
            key, feature_path, label, weight = data
            mat = np.asarray(kaldi_io.read_mat(feature_path))
            self.data_batch.append(
                [torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)])

    def __len__(self):
        return len(self.data_batch)

    def __getitem__(self, idx):
        return self.data_batch[idx]


def pad_tensor(t, pad_to_length, dim):
    pad_size = list(t.shape)
    pad_size[dim] = pad_to_length - t.size(dim)
    return torch.cat([t, torch.zeros(*pad_size).type_as(t)], dim=dim)


class PadCollate:
    def __init__(self, spec_augment=False):
        self._augmentor = None
        if spec_augment:
            self._augmentor = SpecAugmentor(
                freq_mask_param=15, num_freq_mask=2, time_mask_param=23, time_mask_ratio=0.2, num_time_mask=2)

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
        input_batch = map(lambda x: pad_tensor(
            x[0], max_input_length, 0), batch)
        label_batch = map(lambda x: pad_tensor(
            x[1], max_label_length, 0), batch)
        if sys.version > '3':
            input_batch = list(input_batch)
            label_batch = list(label_batch)
        logits = torch.stack(input_batch, dim=0)
        label_padded = torch.stack(label_batch, dim=0)
        input_lengths = torch.IntTensor(input_lengths)
        label_lengths = torch.IntTensor(label_lengths)
        weights = torch.FloatTensor([x[2] for x in batch])
        if self._augmentor is not None:
            logits = self._augmentor(logits)
        return logits, input_lengths, label_padded, label_lengths, weights


class PadCollateChunk:
    def __init__(self, chunk_size=40):
        self.chunk_size = chunk_size

    def __call__(self, batch):
        # batch: list of (mat, label, weight)
        # return: logits, input_lengths, label_padded, label_lengths, weights
        input_lengths = map(lambda x: x[0].size(0), batch)
        if sys.version > '3':
            input_lengths = list(input_lengths)
        max_input_length = max(input_lengths)
        max_input_length = int(
            self.chunk_size*(math.ceil(float(max_input_length)/self.chunk_size)))
        label_lengths = map(lambda x: x[1].size(0), batch)
        if sys.version > '3':
            label_lengths = list(label_lengths)
        max_label_length = max(label_lengths)
        input_batch = map(lambda x: pad_tensor(
            x[0], max_input_length, 0), batch)
        label_batch = map(lambda x: pad_tensor(
            x[1], max_label_length, 0), batch)
        if sys.version > '3':
            input_batch = list(input_batch)
            label_batch = list(label_batch)
        logits = torch.stack(input_batch, dim=0)
        label_padded = torch.stack(label_batch, dim=0)
        input_lengths = torch.IntTensor(input_lengths)
        label_lengths = torch.IntTensor(label_lengths)
        weights = torch.FloatTensor([x[2] for x in batch])
        return logits, input_lengths, label_padded, label_lengths, weights
