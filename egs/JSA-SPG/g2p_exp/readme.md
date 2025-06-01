
# G2P training for SPG-JSA training
Author: Sardar (sar_dar@foxmail.com)

This directory contains the experimental codes for training G2P models that are used for SPG-JSA initialization. The codes are designed to reproduce the experiments described in the paper, covering data preprocessing, model training, and evaluation steps.

> **Tips:** All the codes need to be run in the `<path to CAT>/egs/SPG-JSA` directory.
# Training process
We assume that you have completed the training of the S2P model and the preparation of the data in the CAT style, as described in the [`S2P training guide`](../s2p_exp/readme.md). 

## step 1: generate pseudo-labels from S2P model
* To decode the training set in the S2P folder, you only need to modify `data:test` in the `hyper-p.json` file to the training set and then run the decoding script.

## step 2:  Train a character tokenizer for G2P input
* Configure the following settings in the `hyper-p.json` file:
```json
{
    ...
    "tokenizer": {
            "type": "SentencePieceTokenizer",
            "option-train": {
                "model_type": "char",
                "model_prefix": "data/sentencepiece/id/spm_char",
                "character_coverage": 1.0
            },
            "file": "<lang dir>/tokenizer_char.tknz"
        },
    ...
}
```
* and run the following script for character tokenizer training:
```bash
python utils/pipeline/asr.py <g2p exp dir> --sta 1 --sto 1
```

## step 3:  data packing
* Define a new train/dev set in [`metainfo.json`](../data/metainfo.json). Set `scp` as the phoneme files decoded by S2P, and `trans` as the real texts.
* run the following script for packing data:
```bash
python local/pkl_p2g_data.py <g2p exp dir> <path of character tokenizer> --g2p
```

## step 4:  G2P model training
* You need to add the mapped checkpoint path as the setting for the `train:option:init_model` configuration item within the `hyper-p.json` file.
* run the following script for start the training:
```bash
python utils/pipeline/asr.py <g2p exp dir> --sta 3 --sto 3
```

## step 4:  G2P model evaluation
* Define a new test set in [`metainfo.json`](../data/metainfo.json). Set `trans` as the real texts.
* run the following script for packing data:
```bash
python local/pkl_p2g_data.py <g2p exp dir> <path of character tokenizer> --gpu --test --save2info
```
* run the following script for decoding:
```bash
python utils/pipeline/asr.py <g2p exp dir> --sta 4
```