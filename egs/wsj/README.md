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
| [ctc](exp/asr-ctc-phone)         | phone  | Y   | 6.79   | 11.88 | 13.5         | WFST decode        |
| [ctc-crf](exp/asr-ctc-crf-phone) | phone  | Y   | 2.87   | 5.53  | 13.5         | WFST decode        |
| [rnn-t](exp/asr-rnnt-bpe)        | BPE-2k | Y   | 9.87   | 12.63 | 21.6         | Beam search decode | 


## LM pruned

- Compare experiment result
    
    - I used the kenlm tool to generate a 4-gram language model;and pruned it `--prune 2 2 5 5`

    - In WFST decoding stage;AC score weight and LM score weight `acwt=1.0 lmwt=0.9`

|   Model                          | Unit   | SP  | eval92 | dev93 | \#params (M) | Notes              |
| -------------------------------- | ------ | --- | ------ | ----- | ------------ | ------------------ |
| [ctc](exp/asr-ctc-phone)         | phone  | Y   | 7.77   | 12.69 | 13.5         | WFST decode        |
| [ctc-crf](exp/asr-ctc-crf-phone) | phone  | Y   | 3.67   | 6.62  | 13.5         | WFST decode        |