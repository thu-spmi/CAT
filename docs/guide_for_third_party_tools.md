# Guideline of third-party tools

## Kaldi

- Repository: https://github.com/kaldi-asr/kaldi
- **Kaldi** is required for some FST-related operations and optional for audio feature extraction (alternatively, you can use `torchaudio`).

- We assume the required train/dev/test files satisfy the path format (Kaldi default style):
   ```
   .../data
    ├── <test_1>
    │   ├── feats.scp
    │   └── text
    ├── <test_2>
    │   ├── feats.scp
    │   └── text
    ├── <dev_1>
    │   ├── feats.scp
    │   └── text
    └── <train_1>
        ├── feats.scp
        └── text
   ```
   The `feats.scp`, that looks like as follows, directs to the real features (commonly FBank feat.).
   ```
   TRAIN_001   /path/to/feat.ark:10
   TRAIN_002   /path/to/feat.ark:100
   ...
   ```
   and the `text` file is the transcript, like:
   ```
   TRAIN_001   hello could you recognize me
   TRAIN_002   good morning
   ...
   ```
   Utterences in the `feats.scp` and `text` under the same folder should be matched by the utterance IDs.

- Suppose we want to use features extracted by Kaldi and stored somewhere else, all we need is link the data folder to your working directory as:
   ```bash
   ln -s $(readlink -f /path/to/kaldi/egs/[task]/data) /current/repo/egs/[task]/data/src
   ```
   DON'T FORGET THE `src` AT THE END.


## KenLM

- Repository: https://github.com/kpu/kenlm
- **kenlm** is a tool for training n-gram language model.
- dependencies: you must first ensure your platform satisfies the building requirements of kenlm. See [kenlm/BUILDING](https://github.com/kpu/kenlm/blob/master/BUILDING)