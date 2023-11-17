# copyright 2023 Tsinghua University

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
            
            #waveform = torch.from_numpy(waveform)  
            #解决警告
            
            waveform_copy = waveform.copy()  # 拷贝数组
            waveform = torch.from_numpy(waveform_copy)  # 将拷贝后的数组转换为张量
            
            if processor is not None:
                waveform = processor(waveform)

            #writer(uid, waveform.unsqueeze(1).numpy()) 
            #其中unsqueeze(1)作用是增加一个通道轴，这里不用增加，直接沿用即可
            
            writer(uid, waveform.numpy())

# 未来添加速度扰动，参考cat/utils/data/data_prep.py
"""
    for _factor in speed_perturb:
        if _factor == 1.0:
            continue
        sp_processor = (
            ReadProcessor(normalize=load_with_norm)
            .append(SpeedPerturbationProcessor(_factor, sample_frequency))
            .append(fbank_processor)
        )
        spsuffix = f"#sp{_factor}"
        for _set in subsets:
            try:
                f_trans = fmt_trans.format(f"{_set}-sp{_factor}")
                f_scp = fmt_scp.format(f"{_set}-sp{_factor}")
                f_ark = fmt_ark.format(f"{_set}-sp{_factor}")
                os.makedirs(os.path.dirname(f_trans), exist_ok=True)
                os.makedirs(os.path.dirname(f_scp), exist_ok=True)
                os.makedirs(os.path.dirname(f_ark), exist_ok=True)
                # write trans
                if os.path.isfile(f_trans):
                    sys.stderr.write(f"WARNING: transcript {f_trans} exists, skip.\n")
                else:
                    with open(f_trans, "w") as fo:
                        for uid, utt in trans[_set]:
                            fo.write(f"{uid}{spsuffix}\t{utt}\n")

                # write feats
                if os.path.isfile(f_scp):
                    sys.stderr.write(f"WARNING: scp file {f_scp} exists, skip.\n")
                else:
                    _process_as_kaldi(
                        audios[_set],
                        f_scp,
                        f_ark,
                        sp_processor,
                        uidsuffix=spsuffix,
                        desc=f"{_set} sp {_factor}",
                    )
            except Exception as e:
                if os.path.isfile(f_scp):
                    os.remove(f_scp)
                if os.path.isfile(f_ark):
                    os.remove(f_ark)
                if not read_from_extracted_meta and os.path.isfile(f_trans):
                    os.remove(f_trans)
                raise RuntimeError(str(e))
"""