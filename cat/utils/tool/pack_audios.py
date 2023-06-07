# copyright 2023 Tsinghua University
# author: Huahuan Zheng

"""
pack raw audios to kaldi .ark format, allowing further usage (like wav2vec training.)
"""

import sys
import argparse
import torch
from tqdm import tqdm
from kaldiio import ReadHelper, WriteHelper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wav", type=str, help="File entry of raw audios, usually named as 'wav.scp'"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output file. This should look like 'ark,scp:/path/to/ark,/path/to/scp'",
    )
    parser.add_argument("--segment", type=str, help="Input segments. Default: none.")
    parser.add_argument(
        "--resample",
        type=int,
        default=-1,
        help="Resample to a different sampling rate.",
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip normalization (waveform to (0., 1.)).",
    )
    args = parser.parse_args()

    from cat.utils.data.data_prep import ResampleProcessor, NormalizeProcessor

    wavin = args.wav
    arkout = args.output

    tot = sum(1 for _ in open(wavin, "r"))

    if args.skip_normalize:
        processor = None
    else:
        processor = NormalizeProcessor()

    if args.resample != -1:
        assert args.resample > 0

        src_rate = next(iter(ReadHelper(f"scp:{wavin}", segments=args.segment)))[1][0]
        if src_rate != args.resample:
            sys.stderr.write(f"resample: {src_rate} -> {args.resample}\n")
            _processor1 = ResampleProcessor(src_rate, args.resample)
            processor = (
                _processor1 if processor is None else processor.append(_processor1)
            )

    with ReadHelper(f"scp:{wavin}", segments=args.segment) as reader, WriteHelper(
        arkout
    ) as writer:
        for uid, (rate, waveform) in tqdm(reader, total=tot):
            waveform = torch.from_numpy(waveform)
            if processor is not None:
                waveform = processor(waveform)

            writer(uid, waveform.unsqueeze(1).numpy())
