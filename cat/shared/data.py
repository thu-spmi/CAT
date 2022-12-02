# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Hongyu Xiang,
#         Keyu An,
#         Huahuan Zheng (maxwellzh@outlook.com)

"""Data loading module
"""

from queue import Queue
from . import coreutils as coreutils
from .tokenizer import AbsTokenizer

import io
import os
import kaldiio
import pickle
import math
import hashlib
import numpy as np
from typing import *

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def get_sha256(file: str) -> str:
    '''Get sha256 has of a file.
    '''
    assert os.path.isfile(file), f"{file} not found."
    sha256_hash = hashlib.sha256()
    with open(file, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


class FeatureReader:
    def __init__(self) -> None:
        self._opened_fd = {}

    def __call__(self, arkname: str):
        return kaldiio.load_mat(arkname, fd_dict=self._opened_fd)

    def __del__(self):
        for f in self._opened_fd.values():
            f.close()
        del self._opened_fd


class AbsDataset(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.f_path = path
        assert os.path.isfile(
            path), f"{self.__class__.__name__}: {path} is not a valid file."

    def impl_get_len(self):
        raise NotImplementedError

    def get_seq_len(self) -> List[int]:
        # try to find length info otherwise read from features
        f_linfo = self.f_path+'.linfo'
        if os.path.isfile(f_linfo):
            with open(f_linfo, 'rb') as fi:
                return pickle.load(fi)
        else:
            ls = self.impl_get_len()
            if not os.access(os.path.dirname(f_linfo), os.W_OK):
                print(
                    f"No writing access to: '{f_linfo}'. "
                    "Would reload it at the next time."
                )
            else:
                with open(f_linfo, 'wb') as fo:
                    pickle.dump(ls, fo)
            return ls


class IndexMappingDataset(AbsDataset):
    def __init__(self, f_index: str) -> None:
        super().__init__(f_index)
        self.dataset = None
        with open(f_index, 'rb') as fi:
            self.f_data = os.path.join(os.path.dirname(
                f_index), pickle.load(fi))  # type: str
            if not os.path.isfile(self.f_data):
                raise FileNotFoundError(
                    f"\n{self.__class__.__name__}:\n"
                    f"From indexing file {f_index} mapping to {self.f_data}\n"
                    f"... but {self.f_data} is not found.")
            self.offsets = pickle.load(fi)

    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None

    def impl_get_len(self):
        _ls = np.empty(len(self), dtype=np.int64)
        for i in range(len(self)):
            ''' NOTE (huahuan): 
            suppose `__getitem__` method returns a tuple
            ... where the first item is the feature;
            ... if not the case, impl your custom `impl_get_len` method.
            '''
            x = self[i][0]
            _ls[i] = x.size(0)
        return _ls

    def __len__(self) -> int:
        return len(self.offsets)

    @staticmethod
    def _readbuffer(fileio: "io.BufferedReader"):
        raise NotImplementedError

    def __getitem__(self, index: int):
        if self.dataset is None:
            self.dataset = open(self.f_data, 'rb')
        self.dataset.seek(self.offsets[index], 0)
        # you should impl `_readbuffer` method of your derived class
        return self._readbuffer(self.dataset)


# NOTE (Huahuan):
#    deprecate old speech dataset for better CPU memory efficiency,
#    ... check https://pytorch.org/docs/stable/data.html#multi-process-data-loading
#    ... for why this happened.


class ModifiedSpeechDataset(IndexMappingDataset):
    """Speech dataset"""

    def __init__(self, f_index: str) -> None:
        super().__init__(f_index)

    @staticmethod
    def _readbuffer(fileio: "io.BufferedReader"):
        mat = np.load(fileio)
        label = np.load(fileio)
        return torch.from_numpy(mat), torch.from_numpy(label)


class KaldiSpeechDataset(AbsDataset):
    """Read in kaldi style with ark file.
    
    Data format (store with pickle):
        {
            'label': np.ndarray,
            'linfo': np.ndarray,
            'arkname': np.ndarray,
            'key': np.ndarray
        }
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)
        with open(path, 'rb') as fib:
            self._meta_data = pickle.load(fib)
        self._feat_reader = FeatureReader()

    def filt_by_len(self, filt_func: Callable[[int, int], bool]):
        """filter the dataset according to the `filt_func`, call before loading data.

        filt_func (function): invoked via filt_func(feat_len, label_len), 
            True for keeping the data, False for removed.
        """
        torm = []
        linfo = self._meta_data['linfo']
        labellen = self._meta_data['label'][:, -1]
        for i in range(len(self)):
            if not filt_func(linfo[i], labellen[i]):
                torm.append(i)
        del linfo
        del labellen

        for metakey in ['label', 'linfo', 'arkname', 'key']:
            self._meta_data[metakey] = np.delete(
                self._meta_data[metakey],
                torm, axis=0
            )
        return

    def get_seq_len(self) -> List[int]:
        return self._meta_data['linfo']

    def __len__(self) -> int:
        return len(self._meta_data['linfo'])

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        mat = self._feat_reader(self._meta_data['arkname'][index])
        # remove padding in label
        # [*, *, *, *, -1, -1, rel_len(4)]
        label = self._meta_data['label'][index]
        label = label[:label[-1]]
        return torch.tensor(mat), torch.tensor(label)


class CorpusDataset(IndexMappingDataset):
    """LM corpus dataset"""

    def __init__(self, f_index: str) -> None:
        super().__init__(f_index)

        self._mode = 0
        if len(self) > 0:
            with open(self.f_data, 'rb') as fi:
                fi.seek(self.offsets[0], 0)
                item = pickle.load(fi)
            if isinstance(item, tuple) and len(item) == 2:
                self._mode = 1

    @property
    def mode(self):
        # 1 for return tuple of tensors, 0 for return tensor
        return self._mode

    def _readbuffer(self, fileio: "io.BufferedReader"):
        if self._mode:
            x, y = pickle.load(fileio)
            return torch.LongTensor(x), torch.LongTensor(y)
        else:
            x = pickle.load(fileio)
            return torch.LongTensor(x[:-1]), torch.LongTensor(x[1:])


class ScpDataset(AbsDataset):
    """
    Read data from scp file ranging [idx_beg, idx_end)
    """

    def __init__(self, scp_file: str) -> None:
        super().__init__(scp_file)

        if not os.path.isfile(scp_file):
            raise FileNotFoundError(f"{scp_file} is not a valid file.")

        self._dataset = []
        with open(scp_file, 'r') as fi:
            for line in fi:
                self._dataset.append(line.split())

        self.freader = FeatureReader()

    def __len__(self) -> int:
        return len(self._dataset)

    def impl_get_len(self):
        _ls = []
        for _, fpath in self._dataset:
            mat = self.freader(fpath)
            _ls.append(mat.shape[0])

        del self.freader
        self.freader = FeatureReader()
        return _ls

    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor]:
        key, mat_path = self._dataset[index]
        mat = self.freader(mat_path)
        return [key, torch.tensor(mat, dtype=torch.float)]


class NbestListDataset(AbsDataset):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        with open(self.f_path, 'rb') as fi:
            # type: Dict[str, Dict[int, Tuple[float, str]]]
            self._dataset = list(pickle.load(fi).items())

    def impl_get_len(self):
        return [len(hypos) for _, hypos in self._dataset]

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[List[str], List[float], List[str]]:
        # create new key = nbest id + '-' + original key,
        # so that we can get it back via new_key.split('-', maxsplit=1)
        keys, scores, trans = [], [], []
        okey = self._dataset[index][0]
        for nid, (_score, _trans) in self._dataset[index][1].items():
            keys.append(f"{nid}-{okey}")
            scores.append(_score)
            trans.append(_trans)
        return keys, scores, trans


class NbestListCollate():
    """Collator for N-best list file.
    The passing tokenizer should have method `encode` to convert text to indices.
    """

    def __init__(self, tokenizer: AbsTokenizer, bos_id: int = 0) -> None:
        self._tokenizer = tokenizer
        assert isinstance(
            bos_id, int) and bos_id >= 0, f"ValueError: bos_id={bos_id}"
        self.bos_id = bos_id

    def __call__(self, batches: List[Tuple[List[str], List[float], List[str]]]):
        """
        Args:
            batches : [([key, key, ...], [score1, score2, ...], ["hypo1", "hypo2",...]), ...], length B

        Returns:
            (keys, texts, scores, tokens)
            keys (List[str]): (B * N-best, )
            texts (List[str]): (B * N-best, )
            scores (torch.FloatTensor): (B * N-best, )
            tokens :
            {
                'input_ids' (torch.LongTensor): (B * N-best, L_max)
                'attention_mask' (torch.LongTensor, torch.BoolTensor): (B * N-best, L_max)
            }
        """
        keys, scores, trans = [], [], []
        for lk, ls, lt in batches:
            keys += lk
            scores += ls
            trans += lt

        ids = [[self.bos_id] +
               self._tokenizer.encode(seqs) for seqs in trans]
        token_ids = coreutils.pad_list(
            [torch.LongTensor(i) for i in ids])
        lens = torch.LongTensor([len(x) for x in ids])
        token_mask = torch.arange(
            lens.max())[None, :] >= lens[:, None]

        scores = torch.FloatTensor(scores)
        return keys, trans, scores, token_ids, token_mask


class sortedPadCollateASR():
    """Collect data into batch by desending order according to frame length and add padding.

    Args:
        batch  : list of (mat, label)
        mat    : torch.FloatTensor
        label  : torch.IntTensor

    Return:
        (logits, input_lengths, labels, label_lengths)
    """

    def __init__(self, flatten_target: bool = False) -> None:
        """
        flatten_target (bool): flatten the target to be 1-dim, default False (2-dim)
        """
        self._flatten_target = flatten_target

    def __call__(self, batch: List[Tuple[torch.FloatTensor, torch.IntTensor]]):
        batches = [
            (mat, label, mat.size(0))
            for mat, label in batch
        ]
        batch_sorted = sorted(batches, key=lambda item: item[2], reverse=True)

        mats = coreutils.pad_list([x[0] for x in batch_sorted])

        if self._flatten_target:
            labels = torch.cat([x[1] for x in batch_sorted], dim=0)
        else:
            labels = coreutils.pad_list(
                [x[1] for x in batch_sorted]).to(torch.long)

        input_lengths = torch.LongTensor([x[2] for x in batch_sorted])

        label_lengths = torch.LongTensor([x[1].size(0) for x in batch_sorted])

        return mats, input_lengths, labels, label_lengths


class sortedPadCollateLM():
    """Collect data into batch by desending order and add padding.

    Args:
        batch  : [sentences] or [(labels, targets)] 
            labels  : torch.LongTensor, sentences[:-1]
            targets : torch.LongTensor, sentences[1:]

    Return:
        (labels, label_lengths, targets, `torch.empty(1)`)
    """

    def __init__(self, flatten_target: bool = True) -> None:
        self.flatten_target = flatten_target

    def __call__(self, batch: List[Tuple[torch.LongTensor, torch.LongTensor]]):
        batch_sorted = sorted(
            batch, key=lambda item: item[0].size(0), reverse=True)

        X, Y = list(zip(*batch_sorted))
        xlens = torch.LongTensor([x.size(0)
                                 for x in X])  # type: torch.LongTensor
        ylens = torch.LongTensor([y.size(0) for y in Y])

        xs = coreutils.pad_list(X)   # type: torch.Tensor

        if self.flatten_target:
            target = torch.cat(Y, dim=0)
        else:
            target = coreutils.pad_list(Y)

        return xs, xlens, target, ylens


class sortedScpPadCollate():
    """Collect data into batch and add padding.
    Args:
        batch   : list of (key, feature)
        key     : str
        feature : torch.FloatTensor
    Return:
        (keys, logits, lengths)
    """

    def __call__(self, batch: Sequence[Tuple[str, torch.FloatTensor]]) -> Tuple[Sequence[str], torch.FloatTensor, torch.LongTensor]:

        if len(batch) > 1:
            batch = sorted(
                batch, key=lambda item: item[1].size(0), reverse=True)
        keys = [key for key, _ in batch]

        mats = coreutils.pad_list([feature for _, feature in batch])

        lengths = torch.LongTensor([feature.size(0) for _, feature in batch])

        return keys, mats, lengths


class DynamicBatchDistSampler(DistributedSampler):
    def __init__(self,
                 dataset: AbsDataset,
                 mode: Literal['bucket', 'batch'],
                 global_batch_size: int = -1,
                 max_bucket_size: int = -1,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 local_rank: int = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=drop_last)

        if not hasattr(dataset, 'get_seq_len'):
            raise RuntimeError(
                f"{type(dataset)} has not implement get_seq_len(), "
                f"which is required for {self.__class__.__name__}.")

        # scan data length, this might take a while
        if local_rank is None:
            # using 1 node
            local_rank = self.rank

        if local_rank == 0:
            # save length info into cache file
            dataset.get_seq_len()

        dist.barrier()

        if mode == 'bucket':
            if max_bucket_size == -1:
                linfo = dataset.get_seq_len()
                avglen = sum(linfo)/len(linfo)
                max_bucket_size = int(avglen*global_batch_size)

            assert max_bucket_size > 0 and max_bucket_size >= self.num_replicas, max_bucket_size

            self.index_dispatcher = BucketGrouper(
                max_bucket_size=max_bucket_size,
                rank=self.rank,
                n_procs=self.num_replicas,
                linfo=dataset.get_seq_len()
            )
        elif mode == 'batch':
            assert global_batch_size > 0 and \
                global_batch_size >= self.num_replicas and \
                global_batch_size <= len(dataset), global_batch_size

            self.index_dispatcher = BatchGrouper(
                g_batchsize=global_batch_size,
                rank=self.rank,
                n_procs=self.num_replicas,
                linfo=dataset.get_seq_len()
            )
        else:
            raise ValueError(
                f"{self.__class__.__name__}: unknown mode '{mode}'")

    def __iter__(self):
        # DistributedSampler.__iter__()
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /
                            len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # Add implementation here
        return iter(self.index_dispatcher(indices))


class Grouper:
    def __init__(self, rank: int, n_procs: int, linfo: List[int]) -> None:
        self.weight = linfo
        self.rank = rank
        self.num_procs = n_procs
        self._bsinfo = Queue(maxsize=4096)

    def _call_group(self, indices: List[int]):
        self._bsinfo.put(len(indices))
        g_sorted = sorted(list(zip(
            indices,
            [self.weight[i]for i in indices]
        )), key=lambda x: x[1], reverse=True)
        return coreutils.weighted_group(
            g_sorted, self.num_procs)[self.rank]


class BatchGrouper(Grouper):
    def __init__(self, g_batchsize: int, rank: int, n_procs: int, linfo: List[int]) -> None:
        super().__init__(rank, n_procs, linfo)
        self.g_batchsize = g_batchsize

    def __call__(self, indices: Iterable[int]):
        cur_batch = []
        for idx in indices:
            cur_batch.append(idx)
            if len(cur_batch) == self.g_batchsize:
                yield self._call_group(cur_batch)
                cur_batch.clear()
        return


class BucketGrouper(Grouper):
    def __init__(self, max_bucket_size: int, rank: int, n_procs: int, linfo: List[int]) -> None:
        super().__init__(rank, n_procs, linfo)
        self.max_bucket_size = max_bucket_size

    def __call__(self, indices: Iterable[int]):
        cur_batch = []
        cumsum = 0
        for i in indices:
            cur_batch.append(i)
            cumsum += self.weight[i]
            if cumsum >= self.max_bucket_size and len(cur_batch) >= self.num_procs:
                yield self._call_group(cur_batch)
                cur_batch.clear()
                cumsum = 0
        return


class PipeTokenize:
    def __init__(self, tokenizer: AbsTokenizer) -> None:
        assert isinstance(tokenizer, AbsTokenizer)
        self._tokenizer = tokenizer

    def __call__(self, samples: Tuple[np.ndarray, str]) -> Tuple[np.ndarray, torch.LongTensor]:
        return (torch.as_tensor(samples[0]), torch.LongTensor(self._tokenizer.encode(samples[1])))


class ReadBatchDataLoader:
    def __init__(self, dataloader: DataLoader, bs: int = -1, dynamic: bool = False):
        """
        Args:
            dataloader : any instances of pytorch dataloader
            bs (int)   : global batch size, not required if `dynamic` is True
            dynamic (bool) : tell the use of DynamicBatchDistSampler
        """
        if dynamic:
            assert dataloader.batch_sampler is not None
            assert isinstance(dataloader.batch_sampler,
                              DynamicBatchDistSampler)
            self.batch_size = None
            self._bsinfo = dataloader.batch_sampler.index_dispatcher._bsinfo
        else:
            self.batch_size = bs
            self._bsinfo = None

        self._isdynamic = dynamic
        self.dl = dataloader

    def __iter__(self):
        for batch in self.dl:
            if self._isdynamic:
                bs = self._bsinfo.get()
            else:
                bs = self.batch_size
            yield bs, batch
        return
