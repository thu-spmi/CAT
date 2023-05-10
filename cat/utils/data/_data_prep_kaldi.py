# Copyright 2023 Tsinghua University
# Apache 2.0.
# Author: Huahuan Zheng (maxwellzh@outlook.com)

"""Extract spectrum with kaldi toolkit.

This script allows to pipe kaldi IO to python,
... therefore could be further used with prep_wds.py, etc.

CMVN and speed perturbation is not support yet.

check _extract_feat() for usage.
"""
import os
import sys
import shutil
import kaldiio
from typing import *
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader, IterableDataset

PathLikedObj = str
# if you have a high speed disk, try increase nproc to fasten
# the dataloding
nproc = 16


class IterAudioData(IterableDataset):
    def __init__(self, pip_cmd: str) -> None:
        super().__init__()
        self.rspecifier = pip_cmd

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            pid = 0
            nproc = 1
        else:
            pid = worker_info.id
            nproc = worker_info.num_workers

        with kaldiio.ReadHelper(self.rspecifier) as reader:
            for i, (uid, mat) in enumerate(reader):
                if i % nproc == pid:
                    yield uid, mat
        return


def feat_extractor(  # wav.scp file
    f_wav: PathLikedObj,
    num_mel_bins: int,
    sample_freq: int = None,
):
    kaldibin = f"{os.environ['KALDI_ROOT']}/src/featbin"

    opts = f"--num-mel-bins={num_mel_bins}"
    if sample_freq is not None:
        opts += f" --sample-frequency={sample_freq}"

    segments = os.path.join(os.path.dirname(f_wav), "segments")
    if os.path.isfile(segments):
        sys.stderr.write("segments file exists, using that")
        pipe = (
            f"ark: {kaldibin}/extract-segments scp,p:{f_wav} {segments} ark:- |"
            f"{kaldibin}/compute-fbank-feats {opts} ark:- ark:- |"
        )
    else:
        pipe = f"ark: {kaldibin}/compute-fbank-feats {opts} scp,p:{f_wav} ark:- |"

    dataloader = DataLoader(
        IterAudioData(pipe),
        shuffle=False,
        num_workers=nproc,
        batch_size=None,
        collate_fn=lambda x: x,
    )
    for uid, feat in dataloader:
        yield (uid, feat)
    return


def _extract_feat(
    # wav.scp file
    f_wav: PathLikedObj,
    # output scp file
    f_scp: PathLikedObj,
    # output ark file corresponding to `f_scp`
    f_ark: PathLikedObj,
    num_mel_bins: int,
    sample_freq: int = None,
    desc: str = None,
):
    f_ark = os.path.abspath(f_ark)
    with kaldiio.WriteHelper(f"ark,scp:{f_ark},{f_scp}") as writer:
        for uid, feat in tqdm(
            feat_extractor(f_wav, num_mel_bins, sample_freq), file=sys.stdout, desc=desc
        ):
            writer(uid, feat)


def prepare_kaldi_feat(
    # subsets to be prepared, e.g. ['train', 'dev', 'test']
    subsets: List[str],
    trans: List[PathLikedObj],
    wavs: List[PathLikedObj],
    num_mel_bins: int = 80,
    sample_frequency: Optional[int] = None,
    fmt_scp: str = "data/src/{}/feats.scp",
    fmt_trans: str = "data/src/{}/text",
    fmt_ark: str = "data/src/.kaldi.arks/{}.ark",
):
    assert "KALDI_ROOT" in os.environ, "$KALDI_ROOT not in PATH."
    for i, _set in enumerate(subsets):
        assert os.path.isfile(
            trans[i]
        ), f"transcript for dataset '{_set}' at: '{trans[i]}' not found."
        assert os.path.isfile(
            wavs[i]
        ), f"wav.scp for dataset '{_set}' at: '{wavs[i]}' not found."

        f_trans = fmt_trans.format(_set)
        f_scp = fmt_scp.format(_set)
        f_ark = fmt_ark.format(_set)
        os.makedirs(os.path.dirname(f_trans), exist_ok=True)
        os.makedirs(os.path.dirname(f_scp), exist_ok=True)
        os.makedirs(os.path.dirname(f_ark), exist_ok=True)

        try:
            # write transcript
            if os.path.isfile(f_trans):
                sys.stderr.write(f"WARNING: transcript {f_trans} exists, skip.\n")
            else:
                shutil.copy(trans[i], f_trans)

            # write feats
            if os.path.isfile(f_scp):
                sys.stderr.write(f"WARNING: scp file {f_scp} exists, skip.\n")
            else:
                _extract_feat(
                    wavs[i], f_scp, f_ark, num_mel_bins, sample_frequency, desc=_set
                )
        except Exception as e:
            if os.path.isfile(f_scp):
                os.remove(f_scp)
            if os.path.isfile(f_ark):
                os.remove(f_ark)
            raise RuntimeError(str(e))


if __name__ == "__main__":
    # Example of usage
    prepare_kaldi_feat(
        ["dev"],
        ["data/src/dev/text"],
        ["data/src/dev/wav.scp"],
        sample_frequency=16000,
    )
    pass
