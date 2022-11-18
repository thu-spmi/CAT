## Data

Source: [Common Voice Corpus 11.0](https://commonvoice.mozilla.org/zh-CN/datasets)

### Data preparation

You should first follow the **Common Voice** official guide to download the data.

Then prepare data with:

```
# you can use Common Voice any version data, here I use CV-11.0 by default
# keep your data paths real
bash local/data.sh <data path/language>
```

### Result

Performance is evaluated on CER (%).

130 hours **Chinese(China)** speech data

| model                         | Unit   | SP | dev   | test  |
| ----------------------------- | -----  | -- | ----- | ----  |
| [rnnt](exp/asr-rnnt-chinese/) | char   | N  | 18.14 | 17.14 |


Performance is evaluated on WER (%).

180 hours **Russian** speech data

| model                         | Unit   | SP | dev   | test  |
| ----------------------------- | -----  | -- | ----- | ----  |
| [rnnt](exp/asr-rnnt-russian/) | BPE-2k | N  | 6.44  | 8.55  |
| [ctc](exp/asr-ctc-russian/)   | BPE-2K | N  | 16.22 | 19.50 |
