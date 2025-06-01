# SPG-JSA Training Guide
Author: Sardar (sar_dar@foxmail.com)

This directory contains the experimental codes for SPG-JSA training. The codes are designed to reproduce the experiments described in the paper, covering data preprocessing, model training, and evaluation steps.
> **Tips:** 
> 1. All the codes need to be run in the `<path to CAT>/egs/SPG-JSA` directory.
> 2. `<exp dir>` represents SPG-JSA experiment directory, which contains the `hyper-p.json` and `config.json` files. For example, `jsa_exp/SPG-JSA_indonesian_semi-supervised_100utts`

## 1. SPG-JSA training
### step 1: SPG model initialization
* **S2P model:** For semi-supervised training, fine-tune Whistle (90M) model on 10 minutes of data, follow [S2P training guideline](../s2p_exp/readme).
* **P2G model:** Use the S2P model to generate phoneme pseudo-labels on the full dataset. Then, train the P2G model according to the [P2G training guideline](../p2g_exp/readme.md).
* **G2P model:** Use the S2P model to generate phoneme pseudo-labels on the full dataset. Then, train the G2P model according to the [G2P training guideline](../g2p_exp/readme.md).
* Once the three models are initialized, set the checkpoint path as the `init-model` option in the `config.json` file.
### step 2: tokenization
The three models in the SPG architecture use different tokenizers, all of which are configured in the `hyper-p.json` file. They are as follows: 
* `"tokenizer":"file"`: the path of BPE tokenizer, Used for the output of the P2G model. 
* `"tokenizer":"phone_tokenizer"`: the path of Phoneme tokenizer, Used for the phoneme side of the three models. 
* `"tokenizer":"char_tokenizer"`: the path of Character tokenizer, Used for the input of the G2P model.
### step 3: data packing
* run the following script for packing data:
```bash
python local/pkl_spg_data.py <exp dir>
```
* For semi-supervised training, you need to use the `"tr_set_weight"` parameter in the `hyper-p.json` file to adjust the ratio of the two training sets. For example, `"tr_set_weight": [1, 20]` means that the first dataset configured in `"data":"train"` will be repeated once, and the second dataset (e.g., 100 sentences) will be repeated 20 times. 
### step 4: SPG-JSA training
* If you want to conduct semi-supervised training, configure the text data path of the supervised dataset in the `"jsa:supervised_trans"` parameter of the `config.json` file. Otherwise, run it in an unsupervised manner.
* run the following script for start the training:
```bash
python utils/pipeline/asr.py <exp dir> --sta 3 --sto 3
```

## 2.  SPG-JSA decoding
Currently, GPU decoding is used for MLS decoding, while CPU decoding is used for other types of decoding.
### 2.1 decoding without LM
* Configure the following settings in the `hyper-p.json` file:
```json
{
	...
	"inference": {
		"infer": {
			"bin": "cat.ctc.decode_jsa",
			"option": {
				"beam_size": 16,
				"nj": 16
			}
		},
		...
	},
	...
}
```
* and run the following script:
```bash
python utils/pipeline/asr.py <exp dir> --sta 4
```
> WARNING: The `nj` parameter, similar to the one in Kaldi, represents the number of processes. It's important to note that this parameter impacts the speed of two subsequent steps: **decode with LM** and **MLS rescoring**. Increasing it appropriately can speed up the decoding process. However, setting it too high may lead to process blockage or even system crashes. Please pay attention to the remaining number of processes in your system.
### 2.2 decoding with LM
* Prepare a `<lang dir>` folder. This folder should contain an `<lang dir>/lm` folder, which includes the configuration files `hyper - p.json` and `config.json`. Configure the LM data paths in the `hyper - p.json`.
* run this script to train n-gram language model:
```bash
bash local/lm_decode.sh <exp dir> <lang dir> --sta 1 --sto 1
```
*  run this script to generate language graph:
```bash
bash local/lm_decode.sh <exp dir> <lang dir> --sta 2 --sto 2
```
* run following script to start decoding with n-gram LM. 
* The `lw_range` represents the range of the LM weights. During decoding, all the numbers within this range will be traversed with a step size of 0.1. For example, `0.1,1.0` represents ten numbers from 0.1 to 1.0. If the WER still shows a downward trend, please expand the range of the weight values. 
```bash
bash local/lm_decode.sh <exp dir> <lang dir> --sta 3 --sto 3 --lw_range 0.1,1.0
```
### 2.3 MLS rescoring
* run following script to generate n-best file from WFST decoding。
* `lw` represents the LM weight that performed best among the weights tried in the previous step. For example, if 0.5 gets the lowest WER in the decoding with LM step, then lw will be set to 0.5.
```bash
bash local/lm_decode.sh <exp dir> <lang dir> --sta 4 --sto 4 --lw 0.5
```
* You will find the n-best file in the decoding folder. The path format is like `<exp dir>/decode/<test set>/ac1.0_lm<lw>.n64.nbest`. Copy the path into the `nbest_file` parameter in the `hyper-p.json` file. 
* The `LM_weight` represents the LM weight in MLS decoding, which may be different from the value in the previous step. You need to try different values to obtain the lowest WER.
* The other MLS rescoring configurations are as follows:
```json
{
	...
	"inference": {
		"infer": {
			"bin": "cat.ctc.decode_jsa_mls",
			"option": {
				"nbest_file": "<path of n-best file>",
				"n_samples": 10,
				"LM_weight": 1.0
			}
		},
		...
	},
	...
}
```
* set `CUDA_VISIBLE_DEVICES` and run decoding script:
```bash
python utils/pipeline/asr.py <exp dir> --sta 4
```

### 2.4 evaluate S2P PER
* Configure the following settings in the `hyper-p.json` file:
```json
{
	...
	"inference": {
		"infer": {
			"bin": "cat.ctc.decode_jsa_s2p",
			"option": {
				"beam_size": 16,
				"nj": 16,
			}
		},
		...
	},
	...
}
```
* and run following script:
```bash
python utils/pipeline/asr.py <exp dir> --sta 4
```
## 3. P2G augmentation
### step 1: generate augment data
* Configure the training set in the `data:test` section of the `hyper-p.json` file, and perform S2P decoding on the training set.
* Configure the following settings in the `hyper-p.json` file:
```json
{
	...
	"inference": {
		"infer": {
			"bin": "cat.ctc.decode_jsa_s2p",
			"option": {
				"beam_size": 128,
				"nj": 32,
				"thread_per_woker": 3,
				"save_nbest": true
			}
		}
		...
	},
	...
}
```
* and run the following script:
```bash
python utils/pipeline/asr.py <exp dir> --sta 4
```
### step 2: data packing
* Define a new train/dev set in [`metainfo.json`](../data/metainfo.json). Set `scp` as the phoneme files decoded by S2P, and `trans` as the real texts.
* run the following script for packing data:
* `<p2g aug exp dir>` represents P2G augmentation experiment directory. For example, `p2g_exp/P2G_aug_indonesian_SPG_JSA`
* `<path of n-best file>` represents path of n-best file that produced by step 1.
```bash
python local/pkl_p2g_data.py <p2g aug exp dir> <path of phone tokenizer> --from_nbest_file <path of n-best file>
```
### step 3: P2G training
* Configure the checkpoint of the SPG model for which you want to continue conducting the P2G augmentation experiment in the `train:option:init_model_P2G` configuration item of the `hyper-p.json` file. 
* For other detailed configuration, please refer to `p2g_exp/P2G_aug_indonesian_SPG_JSA`.
* run the following script for start the training:
```bash
python utils/pipeline/asr.py <p2g exp dir> --sta 3 --sto 3
```
### step 4: replace checkpoint
run the following script for replace checkpoint:
```bash
python local/replace_checkpoint.py <path of SPG model> --p2g <path of P2G aug model> --out <new path for the SPG model after P2G augmentation>
```
### step 5: decoding
* In the `hyper-p.json` file in the SPG-JSA experiment directory, add the path of the newly generated checkpoint to the `"resume"` option under `"inference":"option"`. With other configurations remaining unchanged, you can perform **decoding without LM**, **decoding with an LM**, and **MLS rescoring**.

## 4. Language Domain Adaptation (LDA) training 
### step 1: generate phoneme pseudo-labels
* Fill in the cross domain data paths of the training set, validation set, and test set in the `data/metainfo.json` file. Both of the `"scp"`  and `"trans"` options specify the path to the text file.
* run the following script for packing data:
```bash
python local/pkl_p2g_data.py <exp dir> <path of character tokenizer> --gpu --test --save2info
```
* Configure the following settings in the `hyper-p.json` file in SPG-JSA experiment directory:
```json
{
	...
	"data":{
		"test":{
			"train_lda",
			"dev_lda",
			"test_lda"
		}
	},
	...
	"inference": {
		"infer": {
			"bin": "cat.ctc.decode_jsa_g2p",
			"option": {
				"beam_size": 64,
				"nj": 32,
				"thread_per_woker": 3,
				"save_nbest": true
			}
		}
		...
	},
	...
}
```
* and run the following script:
```bash
python utils/pipeline/asr.py <exp dir> --sta 4
```
* Fill in the new data paths of the training set, validation set, and test set in the `data/metainfo.json` file. The `"scp"` option specifies the path to the phoneme file decoded by G2P in the step 1, and the `"trans"` option specifies the path to the original text file.

### step 2～5: 
* Steps 2 to 5 are the same as those in the P2G augmentation training.
