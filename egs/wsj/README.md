# WSJ

80-hour English speech data.

## Data

Prepare data with Kaldi tool. (The data preparation of the WSJ dataset is somewhat complicated, requiring a bunch of scripts. So we just use the recipe from kaldi)

```bash
export KALDI_ROOT=/path/to/kaldi

# get help info
bash local/data_kaldi.sh -h

# prepare data with 3-way speed perturbation (default setting in our experiments)
bash local/data_kaldi.sh /path/to/wsj0 /path/to/wsj1 -use-3way-sp
```

## Results

- Performance is evaluated on WER (%).

| Model                            | Unit   | eval92 | dev93 | \#params (M) | Notes              |
| -------------------------------- | ------ | ------ | ----- | ------------ | ------------------ |
| [ctc](exp/asr-ctc-phone)         | phone  | 6.79   | 11.88 | 13.5         | WFST decode        |
| [ctc-crf](exp/asr-ctc-crf-phone) | phone  | 2.87   | 5.53  | 13.5         | WFST decode        |
| [rnn-t](exp/asr-rnnt-bpe)        | bpe-2k | 9.87   | 12.63 | 21.6         | Beam search decode | 

## Results from CAT-v2

- Performance is evaluated on WER (%).
- SP: 3way speed perturbation.

| Model                                                                 | SP   | Eval92 | Dev93 | Param (M) | Notes                              |
| --------------------------------------------------------------------- | ---- | ------ | ----- | --------- | ---------------------------------- |
| [BLSTM](https://github.com/thu-spmi/CAT/tree/master/egs/wsj/exp/demo) | Y    | 3.65   | 6.30  | 13.49     | -                                  |
| BLSTM                                                                 | N    | 3.90   | 6.24  | 13.49     | from CTC-CRF ICASSP2019            |
| BLSTM                                                                 | Y    | 3.79   | 6.23  | 13.49     | from CTC-CRF ICASSP2019            |
| BLSTM                                                                 | N    | 5.19   | 8.62  | 13.49     | char-based from CTC-CRF ICASSP2019 |
| BLSTM                                                                 | Y    | 5.32   | 8.22  | 13.49     | char-based from CTC-CRF ICASSP2019 |
| VGG-BLSTM                                                             | Y    | 3.2    | 5.7   | 16        | -                                  |
| TDNN-NAS                                                              | Y    | 2.77   | 5.68  | 11.9      | from ST-NAS SLT2021                |

