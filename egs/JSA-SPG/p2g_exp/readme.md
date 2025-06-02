
# P2G training for JSA-SPG training
Author: Sardar (sar_dar@foxmail.com)

This directory contains the experimental codes for training P2G models that are used for JSA-SPG initialization. The codes are designed to reproduce the experiments described in the paper, covering data preprocessing, model training, and evaluation steps.

> **Tips:** All the codes need to be run in the `<path to CAT>/egs/JSA-SPG` directory.
# Training process
We assume that you have completed the training of the S2P model and the preparation of the data in the CAT style, as described in the [`S2P training guide`](../s2p_exp/readme.md). 

## step 1: generate pseudo-labels from S2P model
* To decode the training set in the S2P folder, you only need to modify `data:test` in the `hyper-p.json` file to the training set and then run the decoding script.

## step 2:  tokenizer training
* Train a BPE tokenizer for P2G by running the following script.
```bash
python utils/pipeline/asr.py <p2g exp dir> --sta 1 --sto 1
```

## step 3:  data packing
* Define a new train/dev set in [`metainfo.json`](../data/metainfo.json). Set `scp` as the phoneme files decoded by S2P, and `trans` as the real texts.
* run the following script for packing data:
```bash
python local/pkl_p2g_data.py <p2g exp dir> <path of phone tokenizer>
```

## step 4:  P2G model training
* You need to add the mapped checkpoint path as the setting for the `train:option:init_model` configuration item within the `hyper-p.json` file.
* run the following script for start the training:
```bash
python utils/pipeline/asr.py <p2g exp dir> --sta 3 --sto 3
```

## step 4:  P2G model evaluation
* Define a new test set in [`metainfo.json`](../data/metainfo.json). Set `scp` as the phoneme files decoded by S2P, and `trans` as the real texts.
* run the following script for packing data:
```bash
python local/pkl_p2g_data.py <p2g exp dir> <path of phone tokenizer> --test --input_sqs_from_given --save2info
```
* run the following script for decoding:
```bash
python utils/pipeline/asr.py <p2g exp dir> --sta 4
```