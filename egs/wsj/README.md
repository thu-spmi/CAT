# WSJ

80-hour English speech data.

## Data

Prepare data with Kaldi tool. (The data preparation of the WSJ
dataset is somewhat complicated, requiring a bunch of scripts. So we just use the recipe from kaldi)

```bash
export KALDI_ROOT=/path/to/kaldi

# get help info
bash local/data_kaldi.sh -h

# prepare data with 3-way speed perturbation (default setting in our experiments)
bash local/data_kaldi.sh /path/to/wsj0 /path/to/wsj1 -use-3way-sp
```

## Results

- Performance is evaluated on WER (%).

| Model                            | Unit   | SP  | eval92 | dev93 | \#params (M) | Notes              |
| -------------------------------- | ------ | --- | ------ | ----- | ------------ | ------------------ |
| [ctc](exp/asr-ctc-phone)         | phone  | Y   | 7.09   | 12.00 | 13.5         | WFST decode        |
| [ctc-crf](exp/asr-ctc-crf-phone) | phone  | Y   | 2.68   | 5.57  | 13.5         | WFST decode        |
| [rnn-t](exp/asr-rnnt-bpe)        | BPE-2k | Y   | 13.73  | 16.30 | 21.6         | Beam search decode | 

