## Data
170 hour Mandarin speech data. Mostly reading speech.

**Data prepare**

Use one of the following way:

- Prepare data with `torchaudio`: run following command to get help

   ```bash
   bash local/data.sh -h
   ```

- Prepare data with Kaldi:

   1. You should first have Kaldi tool installed.
   
   2. Get help about how to use Kaldi to prepare data:

      ```bash
      KALDI_ROOT=<path/to/kaldi> bash local/data_kaldi.sh -h
      ```

Source data info will be automatically stored at `data/metainfo.json`. You can run

```bash
cd /path/to/aishell
python utils/data/resolvedata.py
```
to refresh the information. Manually modifying is also OK.

## Result

Data prepare with command:

```bash
bash local/data.sh -sp 1.1 0.9
```

Summarize experiments here.

NOTE: some of the experiments are conduct on previous code base, therefore, the settings might not be compatible to the latest. In that case, you could:

- \[Recommand\] manually modify the configuration files (`config.json` and `hyper-p.json`);

- Checkout to old code base by `hyper-p:commit` info. This could definitely reproduce the reported results, but some modules might be buggy.


**Main results**

Evaluated by CER (%)

| EXP ID              | dev  | test | notes                       |
| ------------------- |:----:|:----:| --------------------------- |
| [rnnt](exp/rnnt-v1) | 3.93 | 4.22 | best result, word lm + LODR |
| [ctc](exp/ctc-v1)   | 4.25 | 4.72 | ctc rescored with word lm   |

**CUSIDE results**

| EXP ID                               | test/streaming  | test/non-streaming | notes                      |
| ------------------------------------ |:---------------:|:------------------:| ---------------------------|
| [rnnt-cuside](exp/rnnt-cuside/)      | 6.02            | 5.12               | -                          |
| [ctc-crf-cuside](exp/ctc-crf-cuside/)| 5.57            | 4.99               | WFST decode with 3-gram lm |

### Ablation study

**LM modeling unit**

CTC model: [LINK](exp/ctc-v1)

The acoustic model is based on Chinese characters. The char-based lm is integrated with shallow fusion, while the word-based one with rescoring.

| Setting                             | dev  | test |
| ----------------------------------- |:----:|:----:|
| no lm                               | 4.65 | 5.21 |
| 5-gram char lm [LINK](exp/lm/lm-v1-char-5gram) | 4.49 | 4.95 |
| 3-gram word lm [LINK](exp/lm/lm-v2-word-3gram) | 4.25 | 4.72 |

RNN-T model: [LINK](exp/rnnt/rnnt-v1)

| Setting                             | dev  | test |
| ----------------------------------- |:----:|:----:|
| no lm                               | 4.43 | 4.76 |
| 5-gram char lm [LINK](exp/lm/lm-v1-char-5gram) | 4.35 | 4.69 |
| 3-gram word lm [LINK](exp/lm/lm-v2-word-3gram) | 4.25 | 4.47 | 


**Feature extraction backends and CMVN**

Performances are reported based on [RNN-T](exp/rnnt/rnnt-v1)

| method                            | dev  | test |
| --------------------------------- | :--: | :--: |
| kaldi prep w/ CMVN by speaker     | 4.44 | 4.80 |
| kaldi prep w/o CMVN               | 4.44 | 4.75 |
| torchaudio w/o CMVN               | 4.43 | 4.76 |
| torchaudio w/ CMVN by utterances  | 4.60 | 5.03 |

It is shown that kaldi/torchaudio without CMVN perform close. Applying CMVN (kaldi) with speaker info does not seem to help. Applying CMVN (torchaudio) by utterance deteriorates the results.

By default, both `local/data.sh` and `local/data_kaldi.sh` do not apply CMVN.
