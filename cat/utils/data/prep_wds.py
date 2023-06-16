# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""Prepare speech data in WebDataset format.

Check __main__ for examples of usage.

P.S.
CMVN is not support yet, which I think is not necessary.
"""


from .data_prep import *
from ._data_prep_kaldi import feat_extractor as kaldiio_feat_extractor
from ..pipeline.common_utils import TextUtterancesOrdered
from ..pipeline._constants import UTTS_PER_FILE

import os
import uuid
import shutil
import kaldiio
import numpy as np
import webdataset as wds
from tqdm import tqdm
from typing import *


import torch
from torch.utils.data import DataLoader, IterableDataset

PathLikedObj = str

# processing setting for pack_data_audio()
nproc = 16


class WdsSink:
    """
    d_out : output folder
    pattern : name pattern of tar files, like '%05d.tar'.
        Don't include a path slash in patttern (like 'path/%05d.tar')
    filters: list of tuples of integers, where
        a tuple include two integers (lo, hi)
        seqs will be grouped according to length invervals
        [lo_0, hi_0), [lo_1, hi_1), ...
        lo = -1, stands for 0
        hi = -1 stands for inf.

    output would be:
    if filters is None
        `d_out/pattern`
    else
        `d_out/lo_0_hi_0/pattern`
        `d_out/lo_1_hi_1/pattern`
        ...
    """

    def __init__(
        self,
        d_out: PathLikedObj,
        pattern: str = "%05d.tar",
        filters: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        if filters is None or len(filters) == 0:
            self._filt = None
            self._indexing_bounds = None
            os.makedirs(d_out, exist_ok=True)
            self._sinks = wds.ShardWriter(
                os.path.join(d_out, pattern), maxcount=UTTS_PER_FILE
            )  # type: Union[wds.ShardWriter, List[wds.ShardWriter]]
            self._cnts = 0
        else:
            self._filt = []  # type: List[Tuple[int, int]]
            for intv in filters:
                assert len(intv) == 2
                lo, hi = intv
                assert isinstance(lo, int)
                assert isinstance(hi, int)
                if lo == -1:
                    lo = 0
                if hi == -1:
                    hi = 1 << 31
                self._filt.append((lo, hi))
            self._filt = sorted(self._filt, key=lambda x: x[0])

            self._sinks = []  # type: Union[wds.ShardWriter, List[wds.ShardWriter]]
            for i, (lo, hi) in enumerate(self._filt):
                if i > 0:
                    assert (
                        lo >= self._filt[i - 1][1]
                    ), f"overlapped interval: {self._filt[i-1]} <-> {(lo, hi)}"

                subdir = f"{lo}_{hi}"
                if hi == (1 << 31):
                    subdir = f"{lo}_INF"
                subdir = os.path.join(d_out, subdir)
                os.makedirs(subdir, exist_ok=True)
                self._sinks.append(
                    wds.ShardWriter(
                        os.path.join(subdir, pattern),
                        maxcount=UTTS_PER_FILE,
                    )
                )
            self._cnts = [0 for _ in range(len(self._filt))]

    def write(self, data: Dict, length: Optional[int] = 0):
        if self._filt is None:
            self._sinks.write(data)
            self._cnts += 1
        else:
            for i, (lo, hi) in enumerate(self._filt):
                if length >= lo and length < hi:
                    self._sinks[i].write(data)
                    self._cnts[i] += 1
                    break

    def __del__(self):
        if not hasattr(self, "_sinks"):
            return
        if self._filt is None:
            self._sinks.close()
        else:
            for i, s in enumerate(self._sinks):
                s.close()
                if self._cnts[i] == 0:
                    shutil.rmtree(os.path.dirname(self._sinks[i].pattern))


class IterAudioData(IterableDataset):
    def __init__(
        self, processor: Processor, iter_audio: Iterable[Tuple[str, PathLikedObj]]
    ) -> None:
        super().__init__()
        self._audio = iter_audio
        assert isinstance(processor, Processor), f"{type(processor)}"
        self._processor = processor

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            pid = 0
            nproc = 1
        else:
            pid = worker_info.id
            nproc = worker_info.num_workers

        for i, (uid, audio) in enumerate(self._audio):
            if i % nproc == pid:
                yield uid, self._processor(audio)
        return


def feat_extractor(
    audios: TextUtterancesOrdered,
    sp: float = None,
    num_mel_bin: int = 80,
    sampling_rate: int = 16000,
) -> Iterable[Tuple[str, np.ndarray]]:
    processor = ReadProcessor(normalize=False)
    if sp is not None and sp != 1.0:
        assert isinstance(sp, (float, int))
        processor.append(SpeedPerturbationProcessor(sp, sampling_rate))
    processor.append(FBankProcessor(sampling_rate, num_mel_bin))
    dataloader = DataLoader(
        IterAudioData(processor=processor, iter_audio=audios),
        # if you have a high speed disk, try increase num_worker to fasten
        # the dataloding
        shuffle=False,
        num_workers=nproc,
        batch_size=None,
    )
    for uid, feat in dataloader:
        yield (uid, feat.numpy())


class FeatExtractor:
    def __init__(
        self,
        f_wavs: Union[PathLikedObj, Iterable[PathLikedObj]],
        speed: float = 1.0,
        num_mel_bins: int = 80,
        sample_rate: int = 16000,
        kaldi_style: bool = False,
    ) -> None:
        self._cache = False
        if kaldi_style:
            assert (
                speed == 1.0
            ), "kaldi IO not support speed perturbation yet. Use utils/data/data_prep_kaldi.sh instead."
            if isinstance(f_wavs, PathLikedObj):
                self._audios = f_wavs
            elif len(f_wavs) == 1:
                self._audios = next(iter(f_wavs))
            else:
                # need to merge the wav.scps
                cache = os.path.join("/tmp", uuid.uuid4())
                with open(cache, "w") as fo:
                    # sort the utterances
                    for uid, utt in TextUtterancesOrdered(f_wavs):
                        fo.write(f"{uid}\t{utt}\n")
                self._cache = True
                self._audios = cache
        else:
            self._audios = TextUtterancesOrdered(f_wavs)
            self._speed = speed
            self._kd = False
        self._sample_rate = sample_rate
        self._num_mel = num_mel_bins
        self._kd = kaldi_style

    def __call__(self):
        return iter(self)

    def __iter__(self):
        if self._kd:
            for uid, feat in kaldiio_feat_extractor(
                self._audios, self._num_mel, self._sample_rate
            ):
                yield uid, feat
        else:
            for uid, feat in feat_extractor(
                self._audios, self._speed, self._num_mel, self._sample_rate
            ):
                yield uid, feat
        return

    def __del__(self):
        if self._cache:
            os.remove(self._audios)


def pack_data_audio(
    reader: FeatExtractor,
    writer: WdsSink,
    transcripts: Union[PathLikedObj, List[PathLikedObj]],
):
    """Extract audios to FBank and parse them with text labels into wds file.

    Args:
        writer (WdsSink) : data writer
        d_set  (str): Path to source data, require `d_set/text` & `d_set/wav.scp`
        sp (float) : speed perturbation, usually 0.9 or 1.1, default: none (1.0)
    """

    if not isinstance(transcripts, list):
        transcripts = [transcripts]

    for _f in transcripts:
        assert os.path.isfile(_f), _f

    transcripts = TextUtterancesOrdered(transcripts)
    for (uidt, trans), (uidm, mat) in tqdm(
        zip(transcripts, reader), total=len(transcripts)
    ):
        assert uidt == uidm, f"uid mismatch: {uidt} != {uidm}"

        # webdataset not allowing period '.' in __key__
        writer.write(
            {
                "__key__": uidt.replace(".", "-"),
                "npy": mat,
                "txt": trans,
            },
            mat.shape[0],
        )


def pack_data_ark(
    writer: WdsSink,
    f_scps: Union[List[PathLikedObj], PathLikedObj],
    f_labels: Union[List[PathLikedObj], PathLikedObj],
):
    """Parsing audio feature (in kaldi .ark) and text label into wds file.

    Args:
        f_scps   (str, list): Kaldi-like-style .scp file(s).
        f_labels (str, list): Pure text file(s) include utterance id and sentence labels. Split by space.
    """

    arks = TextUtterancesOrdered(f_scps)
    texts = TextUtterancesOrdered(f_labels)
    assert len(arks) == len(texts), (
        "pack_data: f_scp and f_label should match on the #lines, "
        f"instead {len(arks)} != {len(texts)}"
    )

    f_opened = {}
    for (uidt, trans), (uida, _ark) in tqdm(zip(texts, arks), total=len(arks)):
        assert uidt == uida, f"uid mismatch: {uidt} != {uida}"
        mat = kaldiio.load_mat(_ark, fd_dict=f_opened)  # type:np.ndarray
        writer.write(
            {
                "__key__": uidt.replace(".", "-"),
                "npy": mat,
                "txt": trans,
            },
            mat.shape[0],
        )

    for f in f_opened.values():
        f.close()


if __name__ == "__main__":
    d_out = f"tmp/example"
    pattern = "%05d.tar"

    # create a data writer
    writer = WdsSink(
        d_out,
        pattern,
        filters=[
            (0, 10),
            (10, 1000),
            (1000, 1200),
            (1500, 2000),
            (2000, 3000),
            (3000, -1),
        ],
    )

    # example of packing data from audios
    feat_reader = FeatExtractor("data/src/dev/wav.scp", kaldi_style=False)
    pack_data_audio(feat_reader, writer, "data/src/dev/text")

    # example of packing data from kaldi .ark file
    pack_data_ark(writer, "data/src/dev/feats.scp", "data/src/dev/text")
