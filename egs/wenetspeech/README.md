## Data

Source: [wenet-e2e/WenetSpeech: A 10000+ hours dataset for Chinese speech recognition (github.com)](https://github.com/wenet-e2e/WenetSpeech)

### Data preparation

You should first follow the WenetSpeech official guide to download the data.

Then prepare data with:

```bash
# this script prepares the 1000-hour train-m subset by default.
# you may manually modify the script if you want to process other subsets.
export KALDI_ROOT=/path/to/kaldi
bash local/data_kaldi.sh
```

### Result

Performance is evaluated on CER (%).

**1000-hour train-m**

| model                         | dev   | test-net | test-meeting | aishell-1 test |
| ----------------------------- | ----- | -------- | ------------ | -------------- |
| [rnnt](exp/train_m/rnnt-v1)   | 11.14 | 12.75    | 20.88        | 7.22           |
| [ctc](exp/train_m/ctc-v1)     | 11.80 | 14.28    | 22.23        | 9.05           |
| [ctc-crf](exp/train_m/crf-v1) | 11.15 | 13.38    | 20.52        | 6.83           |

*The `rnnt` & `ctc` decode without LM.

**10000-hour train-l**

For how to prepare and train model on large dataset, please refer to [this](../../docs/how_to_prepare_large_dataset.md).

| model                          |   dev    | test-net | test-meeting | aishell-1 test |
| ------------------------------ |:--------:|:--------:|:------------:|:--------------:|
| Kaldi                          |   9.07   |  12.83   |    24.72     |      5.41      |
| ESPNet                         |   9.70   | **8.90** |    15.90     |    **3.90**    |
| WeNet                          |   8.88   |   9.70   |    15.59     |      4.61      |
| [RNN-T](exp/train_l/rnnt-v1/)  | **7.82** |   9.32   |  **14.66**   |      5.12      |
| [CTC-CRF](exp/train_l/crf-v1/) |   8.75   |  10.51   |    16.21     |      5.29      |

Kaldi, ESPNet and WeNet results are obtained from https://github.com/wenet-e2e/WenetSpeech#benchmark
