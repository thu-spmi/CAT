"""
Prepare FBank feature for librispeech-960 using torchaudio.
"""


import os
import sys
import glob
import argparse
from typing import List, Dict, Tuple

prepare_sets = [
    'dev-clean',
    'dev-other',
    'test-clean',
    'test-other',
    'train-clean-100',
    'train-clean-360',
    'train-other-500'
]

expect_len = {
    'dev-clean': 2703,
    'dev-other': 2864,
    'test-clean': 2620,
    'test-other': 2939,
    'train-clean-100': 28539,
    'train-clean-360': 104014,
    'train-other-500': 148688
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_data", type=str, default="/data/librispeech/LibriSpeech",
                        help="Directory to source audio files, "
                        f"expect sub-dir: {', '.join(prepare_sets)} in the directory.")

    parser.add_argument("--subset", type=str, nargs='*', default=None,
                        choices=prepare_sets, help="Subset to be processes, default all.")
    args = parser.parse_args()

    process_sets = args.subset
    if process_sets is None:
        process_sets = prepare_sets

    assert os.path.isdir(args.src_data)
    for _set in process_sets:
        assert os.path.isdir(os.path.join(args.src_data, _set)
                             ), f"subset '{_set}' not found in {args.src_data}"

    os.makedirs('data/src', exist_ok=True)
    for _set in process_sets:
        trans = []  # type: List[Tuple[str, str]]
        audios = []  # type: List[Tuple[str, str]]
        d_audio = os.path.join(args.src_data, _set)
        f_audios = glob.glob(f"{d_audio}/**/**/*.flac")

        for f_ in sorted(glob.glob(f"{d_audio}/**/**/*.trans.txt")):
            with open(f_, 'r') as fi:
                for line in fi:
                    uid, utt = line.strip().split(maxsplit=1)
                    trans.append((uid, utt))

        for _raw_wav in f_audios:
            uid = os.path.basename(_raw_wav)
            if uid.endswith('.flac'):
                uid = uid[:-4]
            audios.append((uid, _raw_wav))

        assert len(audios) == len(trans), \
            f"# of audio mismatches # of transcript in {_set}: {len(audios)} != {len(trans)}"
        if len(audios) != expect_len[_set]:
            sys.stderr.write(
                f"warning: found {len(audios)} audios in {_set} subset, but expected {expect_len[_set]}")
        trans = sorted(trans, key=lambda x: x[0])
        audios = sorted(audios, key=lambda x: x[0])
        spk2utt = {}    # type: Dict[str, List[str]]
        for uid, _ in trans:
            sid = '-'.join(uid.split('-')[:-1])
            if sid not in spk2utt:
                spk2utt[sid] = [uid]
            else:
                spk2utt[sid].append(uid)
        # type: List[Tuple[str, List[str]]]
        spk2utt = sorted(spk2utt.items(), key=lambda x: x[0])

        d_dst = f"data/src/{_set}"
        os.makedirs(d_dst, exist_ok=True)
        with open(f"{d_dst}/text", 'w') as fo_text, \
                open(f"{d_dst}/wav.scp", 'w') as fo_wav, \
                open(f"{d_dst}/utt2spk", 'w') as fo_u2s:

            for (uid, t), (uid1, aud) in zip(trans, audios):
                assert uid == uid1, f"UID mismatch: {uid} != {uid1}"
                fo_text.write(f"{uid}\t{t}\n")
                fo_wav.write(f"{uid}\tflac -c -d -s {aud} |\n")
                fo_u2s.write(f"{uid}\t{'-'.join(uid.split('-')[:-1])}\n")

        with open(f"{d_dst}/spk2utt", 'w') as fo_s2u:
            for sid, utts in spk2utt:
                fo_s2u.write(f"{sid}\t{' '.join(utts)}\n")
