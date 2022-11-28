## Data

Source: [Common Voice Corpus](https://commonvoice.mozilla.org)

### Data preparation

You should first follow the **Common Voice** official guide to download the data.

Then prepare data with:

```
# Any version of Common Voice data is OK. Here CV-11.0 is used by default
bash local/data.sh /path/to/data
```

### Result

Performance is evaluated on CER (%).

130 hours **Chinese (China)** speech data

| model                         | Unit   | dev   | test  |
| ----------------------------- | -----  | ----- | ----  |
| [rnnt](exp/asr-rnnt-chinese/) | char   | 18.14 | 17.14 |


Performance is evaluated on WER (%).

180 hours **Russian** speech data

| model                         | Unit   | dev   | test  |
| ----------------------------- | -----  | ----- | ----  |
| [rnnt](exp/asr-rnnt-russian/) | bpe-2k | 6.44  | 8.55  |
| [ctc](exp/asr-ctc-russian/)   | bpe-2K | 16.22 | 19.50 |
