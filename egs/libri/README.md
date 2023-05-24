## Data
960 hour English speech data. Book reading speech.

**Data prepare**

Use one of the options:

- Prepare data with Kaldi (default in results)

   ```bash
   bash local/data_kaldi.sh -h
   ```

- Prepare data with `torchaudio`: run following command to get help

   ```bash
   bash local/data.sh -h
   ```

## Result

Summarize experiments here.

Evaluated by WER (%)

| EXPID                                                              | dev-clean | dev-other | test-clean | test-other |
| ------------------------------------------------------------------ | --------- | --------- | ---------- | ---------- |
| [rnnt](exp/rnnt-v1) + transformer [lm](exp/lm/lm-v1-transformer)   | 1.81      | 4.03      | 1.94       | 4.39       |
| [ctc-crf](exp/crf-v1) + transformer [lm](exp/lm/lm-v1-transformer) | 2.05      | 4.54      | 2.25       | 4.73       |

