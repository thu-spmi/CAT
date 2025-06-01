
# Whistle phoneme FT with 10 minutes of data
Author: Sardar (sar_dar@foxmail.com)

This directory contains the experimental codes for training S2P models that are used for SPG - JSA initialization. The codes are designed to reproduce the experiments described in the paper, covering data preprocessing, model training, and evaluation steps.

> **Tips:** All the codes need to be run in the `<path to CAT>/egs/SPG-JSA` directory.
# Training process

## step 1: prepare data
* **Prepare data in CAT style:** This is a standard step for ASR training using the CAT toolkit. For the CommonVoice dataset, refer to the [CommonVoice recipe](../../../egs/commonvoice/README.md).
* **Prepare phoneme transcription:**  Please follow the Whistle data processing guidelines for [Polish](../../../egs/cv-lang10/lang-process/pl/lang_process.md) or [Indonesian](../../../egs/cv-lang10/lang-process/id/lang_process.md).
* **make 100 utterance (or 10 minutes) of subset:**  for data selection, a `text_phn` file is required , and it should be placed in the training data folder, e.g., `data/pl/excluded_train`. 
```bash
>> head -n 2 data/pl/excluded_train/text_phn
common_voice_pl_20547774        i d u x j ɛ ɡ ɔ u ɲ ɔ ɕ w ɕ ɛ a m ɔ 
common_voice_pl_20547775        k t ɔ ɕ z ɲ ɔ v u ʂ ɛ w v ɛ t͡s t͡s ɛ 
````
* select 100 utterences:
```bash
# select 100 sentences of data
python local/select_seq_via_phn_freq.py data/pl/excluded_train --num_utt 100 --special_phn_list data/pl/excluded_train/special_phn_list --out data/pl/selected_uids

# or select 10 minutes (600 seconds) of data
# python local/select_seq_via_phn_freq.py data/pl/excluded_train --data_duration 600 --special_phn_list data/pl/excluded_train/special_phn_list --out data/pl/selected_uids

cd <path to kaldi>/egs/wsj/s5/
bash utils/subset_data_dir.sh --utt-list data/pl/selected_uids data/pl/excluded_train data/pl/excluded_train_100utts_subset
```

## step 2:  tokenizer training
* You can  follow the Whistle data process guideline and prepare a tiny lexicon use 100 utterances. then you can train a `LexiconTokenizer`.
* or specify the path of `text_phn` file as `trans` in the [`metainfo.json`](../data/metainfo.json) and train a `SimpleTokenizer`.
```bash
python utils/pipeline/asr.py egs/SPG-JSA/s2p_exp/Whistle_phone_ft_polish_100utts --sta 1 --sto 1
```
## step 3:  S2P model training
* Download Whitsle small(90M) model and tokenizer from [here](../../cv-lang10/exp/Multilingual/Multi._phoneme_S/readme.md).
* run the script [`unpack_mulingual_param.py`](../../cv-lang10/local/tools/unpack_mulingual_param.py) to map the phoneme embeddings. 
* You need to add the mapped checkpoint path as the setting for the `train:option:init_model` configuration item within the `hyper-p.json` file.
* run the following script for start the training:
```bash
python utils/pipeline/asr.py egs/SPG-JSA/s2p_exp/Whistle_phone_ft_polish_100utts --sta 2 --sto 3
```
## step 4:  S2P model evaluation
```bash
python utils/pipeline/asr.py egs/SPG-JSA/s2p_exp/Whistle_phone_ft_polish_100utts --sta 4
```