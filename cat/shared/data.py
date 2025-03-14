# Copyright 2022 Tsinghua University
# Apache 2.0.
# Author: Hongyu Xiang,
#         Keyu An,
#         Huahuan Zheng (maxwellzh@outlook.com)

"""Data loading module
"""

from queue import SimpleQueue
from typing import List
from . import coreutils as coreutils
from .tokenizer import AbsTokenizer, load

import io
import os
import bisect
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
    """Get sha256 has of a file."""
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
            path
        ), f"{self.__class__.__name__}: {path} is not a valid file."

    def impl_get_len(self):
        raise NotImplementedError

    def get_seq_len(self) -> List[int]:
        # try to find length info otherwise read from features
        f_linfo = self.f_path + ".linfo"
        if os.path.isfile(f_linfo):
            if os.path.getctime(self.f_path) > os.path.getctime(f_linfo):
                print(
                    f"linfo data at {f_linfo} seems outdated compared to data {self.f_path}. Fallback to update it."
                )
            else:
                with open(f_linfo, "rb") as fi:
                    return pickle.load(fi)

        ls = np.asarray(self.impl_get_len(), dtype=np.int32)
        if not os.access(os.path.dirname(f_linfo), os.W_OK):
            print(
                f"No writing access to: '{f_linfo}'. "
                "Would reload it at the next time."
            )
        else:
            with open(f_linfo, "wb") as fo:
                pickle.dump(ls, fo)
        return ls


class IndexMappingDataset(AbsDataset):
    def __init__(self, f_index: str) -> None:
        super().__init__(f_index)
        self.dataset = None
        with open(f_index, "rb") as fi:
            self.f_data = os.path.join(
                os.path.dirname(f_index), pickle.load(fi)
            )  # type: str
            if not os.path.isfile(self.f_data):
                raise FileNotFoundError(
                    f"\n{self.__class__.__name__}:\n"
                    f"From indexing file {f_index} mapping to {self.f_data}\n"
                    f"... but {self.f_data} is not found."
                )
            self.offsets = pickle.load(fi)

    def __del__(self):
        if hasattr(self, "dataset") and self.dataset is not None:
            self.dataset.close()
            self.dataset = None

    def impl_get_len(self):
        _ls = np.empty(len(self), dtype=np.int64)
        for i in range(len(self)):
            """NOTE (huahuan):
            suppose `__getitem__` method returns a tuple
            ... where the first item is the feature;
            ... if not the case, impl your custom `impl_get_len` method.
            """
            x = self[i][0]
            _ls[i] = x.size(0)

        self.dataset.close()
        self.dataset = None
        return _ls

    def __len__(self) -> int:
        return len(self.offsets)

    @staticmethod
    def _readbuffer(fileio: "io.BufferedReader"):
        raise NotImplementedError

    def __getitem__(self, index: int):
        if self.dataset is None:
            self.dataset = open(self.f_data, "rb")
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
        with open(path, "rb") as fib:
            self._meta_data = pickle.load(fib)
        self._feat_reader = FeatureReader()

    def filt_by_len(self, filt_func: Callable[[int, int], bool]):
        """filter the dataset according to the `filt_func`, call before loading data.

        filt_func (function): invoked via filt_func(feat_len, label_len),
            True for keeping the data, False for removed.
        """
        torm = []
        linfo = self._meta_data["linfo"]
        labellen = self._meta_data["label"][:, -1]
        for i in range(len(self)):
            if not filt_func(linfo[i], labellen[i]):
                torm.append(i)
        del linfo
        del labellen

        for metakey in ["label", "linfo", "arkname", "key"]:
            self._meta_data[metakey] = np.delete(self._meta_data[metakey], torm, axis=0)
        return

    def get_seq_len(self) -> List[int]:
        return self._meta_data["linfo"]

    def __len__(self) -> int:
        return len(self._meta_data["linfo"])

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        mat = self._feat_reader(self._meta_data["arkname"][index])
        # remove padding in label
        # [*, *, *, *, -1, -1, rel_len(4)]
        label = self._meta_data["label"][index]
        label = label[: label[-1]]
        return torch.tensor(mat), torch.tensor(label)


class CorpusDataset(AbsDataset):
    """LM corpus dataset

    check cat/utils/data/pack_corpus.py for how data is packed.
    """

    def __init__(self, f_index: str) -> None:
        super().__init__(f_index)

        path_prefix = os.path.dirname(f_index)
        with open(self.f_path, "rb") as fi:
            # paths to raw data
            self._raw_data = [
                os.path.join(path_prefix, x) for x in pickle.load(fi)
            ]  # type: List[str]
            # lengths of seqs
            self._linfo = pickle.load(fi)  # type: np.ndarray
            # seeks of seqs
            self._seeks = pickle.load(fi)  # type: List[np.ndarray]

        self._read_buffer = [None] * len(
            self._raw_data
        )  # type: List[Union[io.BufferedReader, None]]
        self._cum_len = np.zeros((len(self._raw_data)), dtype=np.int32)
        self._cum_len[0] = self._seeks[0].shape[0]
        for i in range(1, len(self._seeks)):
            self._cum_len[i] = self._seeks[i].shape[0] + self._cum_len[i - 1]

    def __len__(self):
        return self._cum_len[-1]

    def get_seq_len(self) -> List[int]:
        return self._linfo

    def __del__(self):
        if hasattr(self, "_read_buffer"):
            for f in self._read_buffer:
                if f is not None:
                    f.close()

    def __getitem__(self, index: int) -> Any:
        pos = bisect.bisect_right(self._cum_len, index)
        if self._read_buffer[pos] is None:
            self._read_buffer[pos] = open(self._raw_data[pos], "rb")

        if pos > 0:
            index -= self._cum_len[pos - 1]
        self._read_buffer[pos].seek(self._seeks[pos][index], 0)
        data = pickle.load(self._read_buffer[pos])

        if isinstance(data[0], int):
            x = data[:-1]
            y = data[1:]
        else:
            x, y = data

        return torch.LongTensor(x), torch.LongTensor(y)


class ScpDataset(AbsDataset):
    """
    Read data from scp file
    """

    def __init__(self, scp_file: str, sort_by_key: bool = False) -> None:
        super().__init__(scp_file)

        if not os.path.isfile(scp_file):
            raise FileNotFoundError(f"{scp_file} is not a valid file.")

        self._dataset = []
        with open(scp_file, "r") as fi:
            for line in fi:
                self._dataset.append(line.split())
        if sort_by_key:
            self._dataset = sorted(self._dataset, key=lambda x: x[0])
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
    def __init__(self, path: str, sort_by_key: bool = False) -> None:
        super().__init__(path)
        with open(self.f_path, "rb") as fi:
            # type: Dict[str, Dict[int, Tuple[float, str]]]
            if sort_by_key:
                self._dataset = sorted(pickle.load(fi).items(), key=lambda x: x[0])
            else:
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


class NbestListCollate:
    """Collator for N-best list file.
    The passing tokenizer should have method `encode` to convert text to indices.
    """

    def __init__(self, tokenizer: AbsTokenizer, bos_id: int = 0) -> None:
        self._tokenizer = tokenizer
        assert isinstance(bos_id, int) and bos_id >= 0, f"ValueError: bos_id={bos_id}"
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

        ids = [[self.bos_id] + self._tokenizer.encode(seqs) for seqs in trans]
        token_ids = coreutils.pad_list([torch.LongTensor(i) for i in ids])
        lens = torch.LongTensor([len(x) for x in ids])
        token_mask = torch.arange(lens.max())[None, :] >= lens[:, None]

        scores = torch.FloatTensor(scores)
        return keys, trans, scores, token_ids, token_mask


class sortedPadCollateASR:
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
        batches = [(mat, label, mat.size(0)) for mat, label in batch]
        batch_sorted = sorted(batches, key=lambda item: item[2], reverse=True)

        mats = coreutils.pad_list([x[0] for x in batch_sorted])

        if self._flatten_target:
            labels = torch.cat([x[1] for x in batch_sorted], dim=0)
        else:
            labels = coreutils.pad_list([x[1] for x in batch_sorted]).to(torch.long)

        input_lengths = torch.LongTensor([x[2] for x in batch_sorted])

        label_lengths = torch.LongTensor([x[1].size(0) for x in batch_sorted])

        return mats, input_lengths, labels, label_lengths


class sortedPadCollateLM:
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
        batch_sorted = sorted(batch, key=lambda item: item[0].size(0), reverse=True)

        X, Y = list(zip(*batch_sorted))
        xlens = torch.LongTensor([x.size(0) for x in X])  # type: torch.LongTensor
        ylens = torch.LongTensor([y.size(0) for y in Y])

        xs = coreutils.pad_list(X)  # type: torch.Tensor

        if self.flatten_target:
            target = torch.cat(Y, dim=0)
        else:
            target = coreutils.pad_list(Y)

        return xs, xlens, target, ylens


class sortedScpPadCollate:
    """Collect data into batch and add padding.
    Args:
        batch   : list of (key, feature)
        key     : str
        feature : torch.FloatTensor
    Return:
        (keys, logits, lengths)
    """

    def __call__(
        self, batch: Sequence[Tuple[str, torch.FloatTensor]]
    ) -> Tuple[Sequence[str], torch.FloatTensor, torch.LongTensor]:
        if len(batch) > 1:
            batch = sorted(batch, key=lambda item: item[1].size(0), reverse=True)
        keys = [key for key, _ in batch]

        mats = coreutils.pad_list([feature for _, feature in batch])

        lengths = torch.LongTensor([feature.size(0) for _, feature in batch])

        return keys, mats, lengths


class BatchDistSampler(DistributedSampler):
    def __init__(
        self,
        dataset: AbsDataset,
        mode: Literal["bucket", "batch"] = "batch",
        dispatch_even: bool = False,
        global_batch_size: int = -1,
        max_bucket_size: int = -1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: int = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        if self.num_replicas == 1:
            dispatch_even = True

        if not (dispatch_even and mode == "batch"):
            # get length info
            if not hasattr(dataset, "get_seq_len"):
                raise RuntimeError(
                    f"{type(dataset)} has not implement get_seq_len(), "
                    f"which is required for {self.__class__.__name__}."
                )

            # scan data length, this might take a while
            if local_rank is None:
                # using 1 node
                local_rank = self.rank

            if local_rank == 0:
                # save length info into cache file
                dataset.get_seq_len()

            dist.barrier()

        self.g_bs = global_batch_size

        if mode == "bucket":
            linfo = dataset.get_seq_len()
            avglen = sum(linfo) / len(linfo)
            if max_bucket_size == -1:
                max_bucket_size = int(avglen * global_batch_size)
            else:
                self.g_bs = int(max_bucket_size // avglen)

            assert (
                max_bucket_size > 0 and max_bucket_size >= self.num_replicas
            ), max_bucket_size

            self.index_dispatcher = BucketGrouper(
                max_bucket_size=max_bucket_size,
                rank=self.rank,
                n_procs=self.num_replicas,
                linfo=linfo,
                dispatch_even=dispatch_even,
            )
        elif mode == "batch":
            assert (
                global_batch_size > 0
                and global_batch_size >= self.num_replicas
                and global_batch_size <= len(dataset)
            ), global_batch_size

            self.index_dispatcher = BatchGrouper(
                g_batchsize=global_batch_size,
                rank=self.rank,
                n_procs=self.num_replicas,
                linfo=(None if dispatch_even else dataset.get_seq_len()),
                dispatch_even=dispatch_even,
            )
        else:
            raise ValueError(f"{self.__class__.__name__}: unknown mode '{mode}'")

    def __len__(self) -> int:
        """Roughly estimated value, might be incorrect."""
        return len(self.dataset) // self.g_bs

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
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # Add implementation here
        return iter(self.index_dispatcher(indices))


class Grouper:
    def __init__(
        self,
        rank: int,
        n_procs: int,
        linfo: List[int] = None,
        dispatch_even: bool = False,
    ) -> None:
        if not dispatch_even:
            assert (
                linfo is not None
            ), "when dispatching batches unevenly, length info is required."
        self.linfo = linfo
        self.rank = rank
        self.num_procs = n_procs
        self.dispatch_even = dispatch_even
        # NOTE (huahuan): use a infinite size queue,
        # otherwise the dataloader might be blocked at put() method
        self._bsinfo = SimpleQueue()

    def _call_group(self, indices: List[int]) -> List[int]:
        assert len(indices) >= self.num_procs
        self._bsinfo.put(len(indices))
        if self.dispatch_even:
            return indices[self.rank : len(indices) : self.num_procs]
        else:
            g_sorted = sorted(
                list(zip(indices, [self.linfo[i] for i in indices])),
                key=lambda x: x[1],
                reverse=True,
            )
            return coreutils.weighted_group(g_sorted, self.num_procs)[self.rank]


class BatchGrouper(Grouper):
    def __init__(self, g_batchsize: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.g_batchsize = g_batchsize

    def __call__(self, indices: Iterable[int]):
        cur_batch = []
        for i in indices:
            cur_batch.append(i)
            if len(cur_batch) == self.g_batchsize:
                yield self._call_group(cur_batch)
                cur_batch.clear()
        return


class BucketGrouper(Grouper):
    def __init__(self, max_bucket_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_bucket_size = max_bucket_size

    def __call__(self, indices: Iterable[int]):
        cur_batch = []
        cumsum = 0
        for i in indices:
            sample_size = self.linfo[i]  # 读取当前样本的大小
            if cumsum + sample_size > self.max_bucket_size and len(cur_batch) >= self.num_procs:
                # 如果添加当前样本会导致超过 max_bucket_size 或者当前bucket溢出，
                # 则返回当前bucket，并将当前样本放入下一个bucket
                yield self._call_group(cur_batch)
                cur_batch.clear()
                cumsum = 0
            cur_batch.append(i)
            cumsum += sample_size

        # 处理剩余的数据，如果有的话
        if cur_batch:
            yield self._call_group(cur_batch)
        
        return


class PipeTokenize:
    def __init__(self, tokenizer: AbsTokenizer) -> None:
        assert isinstance(tokenizer, AbsTokenizer)
        self._tokenizer = tokenizer

    def __call__(
        self, samples: Tuple[np.ndarray, str]
    ) -> Tuple[np.ndarray, torch.LongTensor]:
        return (
            torch.as_tensor(samples[0]),
            torch.LongTensor(self._tokenizer.encode(samples[1])),
        )


class ReadBatchDataLoader:
    """Get batch with the batch size, used with BatchDistSampler and Grouper."""

    def __init__(
        self, dataloader: DataLoader, bs: int = -1, collect_via_dist: bool = False
    ):
        """
        Args:
            dataloader : any instances of pytorch dataloader
            bs (int)   : global batch size, not required if dataloader is BatchDistSampler
            collect_via_dist (bool): collect batchsize via ddp all_reduce, this should be the last
                        choice if the others do not work.
        """
        if collect_via_dist:
            assert dist.is_initialized()
            self.dist_collect = True
            self._bs_buffer = torch.tensor([0], device=torch.cuda.current_device())
            self._sync_stream = torch.cuda.Stream()
        else:
            self.dist_collect = False
            assert isinstance(bs, int)
            if bs > 0:
                self.g_bs = bs
                self._bsinfo = None
            else:
                assert dataloader.batch_sampler is not None
                assert isinstance(dataloader.batch_sampler, BatchDistSampler)
                self._bsinfo = dataloader.batch_sampler.index_dispatcher._bsinfo

                self.g_bs = -1
        self.dl = dataloader

    def get_bs(self) -> int:
        """Call this function every update. Otherwise returned value might be wrong."""
        if self.dist_collect:
            torch.cuda.default_stream().wait_stream(self._sync_stream)
            return self._bs_buffer.item()
        elif self.g_bs == -1:
            return self._bsinfo.get()
        else:
            return self.g_bs

    def __len__(self) -> int:
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            if self.dist_collect:
                with torch.cuda.stream(self._sync_stream):
                    # assume batch is a tuple, where the first element is (N, ...)
                    self._bs_buffer.fill_(batch[0].shape[0])
                    handle = dist.all_reduce(
                        self._bs_buffer, op=dist.ReduceOp.SUM, async_op=True
                    )
                    handle.wait()
            yield batch
        return


# dynamic batching for webdataset
# refer to https://github.com/webdataset/webdataset/blob/main/webdataset/filters.py#L466
class PipeDynamicBatching:
    """Batching with given bucketsize for dynamic batching.
    Apply this after .decode() ; This would drop the last res.

    node_bucket_size: target batch size
    collate_fn: collation function to gather the batch
    count_paddings: count the seqs and the paddings

        e.g.
        A list of data with lengths [1, 1, 1, 1, 1, 995] sums up to 1000 (overall bucket size).

        However, if we conduct padding, the lengths is indeed 995*6=4975, probably causing CUDA OOM.
    """

    def __init__(
        self,
        node_bucket_size: int,
        collate_fn: Optional[Callable] = None,
        count_paddings: bool = True,
    ) -> None:
        assert isinstance(node_bucket_size, int)
        assert node_bucket_size > 0
        assert isinstance(count_paddings, bool)

        self.node_bucket_size = node_bucket_size
        self.collate_fn = collate_fn
        self.count_paddings = count_paddings

    def __call__(self, data: Iterator) -> Iterator:
        if self.count_paddings:
            return self.count_padding_impl(data)
        else:
            return self.native_impl(data)

    def native_impl(self, data: Iterator) -> Iterator:
        batch = []
        cnt = 0
        # sample: (mat, text)
        for sample in data:
            cnt += sample[0].shape[0]
            batch.append(sample)
            if cnt >= self.node_bucket_size:
                if self.collate_fn is None:
                    yield batch
                else:
                    yield self.collate_fn(batch)
                batch.clear()
                cnt = 0
        return

    def count_padding_impl(self, data: Iterator) -> Iterator:
        batch = []
        max_bin = 0
        for sample in data:
            max_bin = max(max_bin, sample[0].shape[0])
            if max_bin * len(batch) + max_bin > self.node_bucket_size:
                assert len(batch) > 0
                if self.collate_fn is None:
                    yield batch
                else:
                    yield self.collate_fn(batch)
                batch.clear()
                max_bin = sample[0].shape[0]
            batch.append(sample)
        return

class WeightedConcatDataset(Dataset):
    def __init__(
        self, datasets: List[AbsDataset], weights: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        self._datasets = datasets
        if weights is None:
            self._weights = [1] * len(self._datasets)
        else:
            for w in weights:
                assert (
                    isinstance(w, int) and w > 0
                ), f"weight must be positive integer, instead {weights}"
            self._weights = weights
        assert len(self._datasets) == len(self._weights)
        self.num_subdatasets = len(self._datasets)

        lengths = [
            len(self._datasets[i]) * self._weights[i]
            for i in range(self.num_subdatasets)
        ]
        for k in range(1, self.num_subdatasets):
            lengths[k] += lengths[k - 1]
        self._indexing_intervals = lengths

    def __len__(self) -> int:
        return sum(
            len(self._datasets[i]) * self._weights[i]
            for i in range(self.num_subdatasets)
        )

    def get_seq_len(self) -> List[int]:
        out = np.empty((len(self),), dtype=np.int32)
        offset = 0
        for i in range(self.num_subdatasets):
            subset_lens = self._datasets[i].get_seq_len()
            for _ in range(self._weights[i]):
                if isinstance(subset_lens, list):
                    subset_lens = np.asarray(subset_lens)
                out[offset : offset + len(self._datasets[i])] = subset_lens
                offset += len(self._datasets[i])

        return out

    def __getitem__(self, index: int):
        isubset = bisect.bisect_right(self._indexing_intervals, index)
        if isubset > 0:
            index -= self._indexing_intervals[isubset - 1]
        index %= len(self._datasets[isubset])
        return self._datasets[isubset][index]



class JSASpeechDataset(AbsDataset):
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
        with open(path, "rb") as fib:
            self._meta_data = pickle.load(fib)
        self._feat_reader = FeatureReader()

    def filt_by_len(self, filt_func: Callable[[int, int], bool]):
        """filter the dataset according to the `filt_func`, call before loading data.

        filt_func (function): invoked via filt_func(feat_len, label_len),
            True for keeping the data, False for removed.
        """
        torm = []
        linfo = self._meta_data["linfo"]
        labellen = self._meta_data["label1"][:, -1]
        for i in range(len(self)):
            if not filt_func(linfo[i], labellen[i]):
                torm.append(i)
        del linfo
        del labellen

        for metakey in ["label1", "label2", "linfo", "arkname", "key"]:
            self._meta_data[metakey] = np.delete(self._meta_data[metakey], torm, axis=0)
        return

    def get_seq_len(self) -> List[int]:
        return self._meta_data["linfo"]

    def __len__(self) -> int:
        return len(self._meta_data["linfo"])

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        mat = self._feat_reader(self._meta_data["arkname"][index])
        # remove padding in label
        # [*, *, *, *, -1, -1, rel_len(4)]
        label = self._meta_data["label1"][index]
        label = label[: label[-1]]
        extra_data = self._meta_data["label2"][index]
        extra_data = extra_data[: extra_data[-1]]
        key = self._meta_data["key"][index]
        return torch.tensor(mat), torch.tensor(label), torch.tensor(extra_data), key

class JSAsortedPadCollateASR:
    """Collect data into batch by desending order according to frame length and add padding.

    Args:
        batch  : list of (mat, label)
        mat    : torch.FloatTensor
        label  : torch.IntTensor

    Return:
        (logits, input_lengths, labels, label_lengths, extras, extras_lengths, indexs)
    """

    def __init__(self, flatten_target: bool = False) -> None:
        """
        flatten_target (bool): flatten the target to be 1-dim, default False (2-dim)
        """
        self._flatten_target = flatten_target

    def __call__(self, batch: List[Tuple[torch.FloatTensor, torch.IntTensor]]):
        batches = [(mat, label, mat.size(0), extra, key) for mat, label, extra, key in batch]
        batch_sorted = sorted(batches, key=lambda item: item[2], reverse=True)

        mats = coreutils.pad_list([x[0] for x in batch_sorted])

        if self._flatten_target:
            labels = torch.cat([x[1] for x in batch_sorted], dim=0)
            extras = torch.cat([x[3] for x in batch_sorted], dim=0)
        else:
            labels = coreutils.pad_list([x[1] for x in batch_sorted]).to(torch.long)
            extras = coreutils.pad_list([x[3] for x in batch_sorted]).to(torch.long)

        input_lengths = torch.LongTensor([x[2] for x in batch_sorted])

        label_lengths = torch.LongTensor([x[1].size(0) for x in batch_sorted])

        extras_lengths = torch.LongTensor([x[3].size(0) for x in batch_sorted])

        indexs = [x[4] for x in batch_sorted]

        return mats, input_lengths, labels, label_lengths, extras, extras_lengths, indexs

class P2GDataset(AbsDataset):
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
        with open(path, "rb") as fib:
            self._meta_data = pickle.load(fib)
        
    def filt_by_len(self, filt_func: Callable[[int, int], bool]):
        """filter the dataset according to the `filt_func`, call before loading data.

        filt_func (function): invoked via filt_func(feat_len, label_len),
            True for keeping the data, False for removed.
        """
        torm = []
        linfo = self._meta_data["linfo"]
        labellen = self._meta_data["label"][:, -1]
        filt_func = lambda x, y: x > y + 2
        for i in range(len(self)):
            if not filt_func(linfo[i], labellen[i]):
                torm.append(i)
        del linfo
        del labellen
        for metakey in ["label", "linfo", "arkname", "key"]:
            self._meta_data[metakey] = np.delete(self._meta_data[metakey], torm, axis=0)
        return

    def get_seq_len(self) -> List[int]:
        return self._meta_data["linfo"]

    def __len__(self) -> int:
        return len(self._meta_data["linfo"])

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        mat = self._meta_data["arkname"][index]
        mat = mat[: mat[-1]]
        label = self._meta_data["label"][index]
        label = label[: label[-1]]
        return torch.tensor(mat), torch.tensor(label)

class P2GLargeDataset(AbsDataset):

    # def __init__(self, path: str, bpe_tokenizer_file:AbsTokenizer, phone_tokenizer_file: AbsTokenizer) -> None:
    def __init__(self, path: str):
        super().__init__(path)
        with open(path, "rb") as fib:
            self._meta_data = pickle.load(fib)
        
        # phone_tokenizer = load(phone_tokenizer_file)
        # self.phone_list = phone_tokenizer._units
        # self.bpe_tokenizer = load(bpe_tokenizer_file)
        
    def filt_by_len(self, filt_func: Callable[[int, int], bool]):
        """
        WARNING: if you using 'P2GLargeDataset', we assume you already done filt_by_len process in pkl data step.
        """
        # print(f"### WARNING: if you using 'P2GLargeDataset', we assume you already done filt_by_len process in pkl data step.")

        return

    def get_seq_len(self) -> List[int]:
        return self._meta_data["in_len"]

    def __len__(self) -> int:
        return len(self._meta_data["key"])

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        # mat = [self.phone_list[phone] for phone in self._meta_data["input"][index].split()]
        # label = self.bpe_tokenizer.encode(self._meta_data["label"][index])
        mat = self._meta_data["input"][index]
        label = self._meta_data["label"][index]
        return torch.tensor(mat, dtype=torch.int64), torch.tensor(label, dtype=torch.int64)

class P2GTestDataset(AbsDataset):
    """
    Read data from text_phn file
    """

    def __init__(self, scp_file: str, sort_by_key: bool = False) -> None:
        super().__init__(scp_file)

        if not os.path.isfile(scp_file):
            raise FileNotFoundError(f"{scp_file} is not a valid file.")

        self._dataset = []
        with open(scp_file, "r") as fi:
            for line in fi:
                try:
                    uid, phn = line.strip().split(maxsplit=1)
                except ValueError:
                    if len(line) != 0:
                        uid = line.strip().split(maxsplit=1)[0]
                        phn = '0'
                        print(f"### Warning: {uid} has empty transcript!")
                phn = list(map(int, phn.split()))
                self._dataset.append([uid, phn])
        if sort_by_key:
            self._dataset = sorted(self._dataset, key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self._dataset)

    def impl_get_len(self):
        _ls = []
        for _, mat in self._dataset:
            _ls.append(len(mat))
        return _ls

    def __getitem__(self, index: int) -> Tuple[str, torch.FloatTensor]:
        key, mat = self._dataset[index]
        return [key, torch.tensor(mat)]