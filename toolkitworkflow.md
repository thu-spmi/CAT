# Toolkit Workflow

We may have different state topologies in the CRF-based ASR framework. In the following, we take phone-based WSJ experiment as an example to illustrate the **step-by-step workflow** of running [CTC-CRF](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ctc-crf.pdf) (namely CRF with CTC topology), which has achieved superior benchmarking performance. Character-based workflow is similar. Scripts from other toolkits are acknowledged.

To begin, go to an example directory under the `egs` directory, e.g. `egs/wsj`, and **run.sh** is the top script, which consists of the following steps.

1. [Data preparation](#Data-preparation)
2. [Feature extraction](#Feature-extraction)
3. [Denominator LM preparation](#Denominator-LM-preparation)
4. [Neural network training preparation](#Neural-network-training-preparation)
5. [Model training](#Model-training)
6. [Decoding](#Decoding)

### Data preparation

**1)** `local/wsj_data_prep.sh` from Kaldi

Do data preparation.  When completed, the folder `data/train` should contain following files:

```
spk2gender
spk2utt
text
utt2spk
wav.scp
```

**2) `local/wsj_prepare_phn_dict.sh`** from Eesen

Download lexicon files and save in folder `data/local/dic_phn`.

```
units.txt : used to generate T.fst (a WFST representation of the CTC topology) later.
lexicon.txt : used to generate L.fst (a WFST representation of the lexicon) later.
```

**3) `ctc-crf/ctc_compile_dict_token.sh`** from Eesen

Compile `T.fst` and `L.fst`.

Note that Eesen `T.fst` (created by `utils/ctc_token_fst.py` in Eesen) makes mistakes, as described in the [CTC-CRF paper](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ctc-crf.pdf). We correct it by a new `scripts/ctc-crf/ctc_token_fst_corrected.py`, which is called by `ctc_compile_dict_token.sh` to create the correct `T.fst`.

**4) `local/wsj_format_local_lms.sh`** from Kaldi

Complie `G.fst` (a WFST representation of the LM used later in ASR decoding) and save in `lang_phn_test_{suffix}`.  The fields in {suffix} could be: tg (3-gram), fg (4-gram), pr (pruned LM), and const (ConstArpa-type LM).

**5) `local/wsj_decode_graph.sh`** from Eesen

Compose `T.fst`、`L.fst`、`G.fst` into `TLG.fst`, which is placed in folder `lang_phn_test_{suffix}`.



Summary of [Data preparation](#Data-preparation): 
![TLG](assets/TLG.png)

### Feature extraction

**1) `utils/subset_data_dir_tr_cv.sh`** from Kaldi

Split train set and dev set in folder `data`. There are two options to split, according to speakers or utterances respectively, configured by `--cv-spk-percent` or `--cv-utt-percent` respectively.

**2) `utils/data/perturb_data_dir_speed_3way.sh`** from kaldi

3-fold data augmentation by perturbing the speaking speed of the original training speech data. The augmented data are postfixed with `sp`,  so as to be differentiated from the original data.

**3) `steps/make_fbank.sh`** from kaldi

Extract filter bank features, and place in folder `fbank`.

**4) `steps/compute_cmvn_stats.sh`** from Kaldi

Compute the mean and variance of features for feature normalization.

### Denominator LM preparation

**1) `ctc-crf/prep_ctc_trans.py`** from Eesen

The training transcripts are saved in `text` file. Based on lexicon, convert word sequences  in `text` file to label sequences and place in `text_number` file. For example,

```
IT GAVE ME THE FEELING I WAS PART OF A LARGE INDUSTRY 
```

will be converted to

```
38 59 35 32 67 46 41 24 9 34 41 45 37 48 19 68 4 70 55 4 56 59 10 67 9 45 4 56 43 38 47 23 9 57 59 56 40
```

**2) `chain-est-phone-lm`** from Kaldi

Sort the training transcripts in `text_number` file according to head labels in label sequences, remove identical label sequences, and obtain `unique_text_number` file.

Based on `unique_text_number` file, train a phone-based language model `phone_lm.fst` and place in folder `data/den_meta`.

**3) `ctc-crf/ctc_token_fst_corrected.py`**

Create the correct `T_den.fst`.

**4) `fstcompose`** from Kaldi

Compose `phone_lm.fst` and `T_den.fst` to `den_lm.fst`, and place in folder `data/den_meta`.

Summary of [Denominator LM preparation](#Denominator-LM-preparation): 
  ![den](assets/den.png)

### Neural network training preparation

For train set, dev set and test set, do the following steps respectively.

**1) apply-cmvn** from Kaldi

Apply feature normalization to the input feature sequence, write to `feats.scp`.

**2) add-deltas** from Kaldi

Calculate the delta features for the input feature sequence.

**3) subsample-feats** from Kaldi

Sub-sample the input feature sequence (default sampling rate: 3).

**4) `path_weight/build/path_weight`**

Note that the potential function (as shown in the [CTC-CRF paper](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ctc-crf.pdf))

<img src="assets/potential.png" alt="potential" width="294.4" height="68">

consists of the denominator LM weight for each training utterance, in addition to the log-softmax weights from the bottom neural neural network.  We need to calculate and save the weight for the label sequence, by the following steps:

- Construct a `linearFST` for each label sequence in `text_number` file;
- Compose the `linearFST` with `phone_lm.fst` to obtain `ofst`.
- Calculate the path weight from `ofst`. 

**5) ctc-crf/convert_to.py**

Save features, `text_number`, and the corresponding path weights into folder `data/hdf5`or`data/pickle`. This file is used as the input of neural network training.

### Model training

**1)  Configuration**

Refer to our examples: `CAT/egs/wsj/exp/demo` and `CAT/egs/wsj/exp/demo2` for enabling SpecAug.

**2) Neural network definition**

The definition of our neural network is in **model.py**. The default models in our demos are BLSTM. 

**3) Loss function**

The output of BLSTM is passed through a fully-conneted layer and a log-softmax layer, which is then used together with the labels to calculate the following loss[^loss] --- Eq (4) in the [CAT paper](https://arxiv.org/abs/1911.08747), by `class CTC_CRF_LOSS` in **ctc_crf.py**.

<img src="assets/loss.png" alt="loss" width="326.5" height="70">

[^loss]: As convention, loss is the negative of log-likelihood.

Note that in the python code, the path weights are not included in the loss for back-propagation because they behave as constants during back-propagation, so we call the loss `partial_loss` for sake of clarity.

The loss function is defined by `class CTC_CRF_LOSS` in **ctc_crf.py**, which calls two functions --- `gpu_ctc` (for the numerator `costs_ctc` calculation) and `gpu_den` (for the denominator `costs_alpha_den` calculation, including  weights for all possible paths). Both functions are implemented with CUDA. The interface definitions for the two functions are in `src/ctc_crf/binding.cpp` and `src/ctc_crf/binding.h`, and the implementations are in `src/ctc_crf/gpu_den` and `src/ctc_crf/gpu_ctc`. For the numerator calculation, we borrowed some codes from [warp-ctc](https://github.com/baidu-research/warp-ctc)。

`costs_ctc` and `costs_alpha_den` are used to calculate the `partial_loss` as follows: 

```
partial_loss = (- costs_ctc + costs_alpha_den) - lamb * costs_ctc
```

where `lamb` is the weight for the CTC Loss, which is employed to stabilize the training.

### Decoding

**1) calculate_logits.py**

Do inference over the test set, using the trained model. The outputs of the network are saved in the format of ark files in folder `exp/demo/logits`.

**2) decode.sh**

Consists of two steps : **latgen-faster** and **score.sh**:

- **latgen-faster** from Eesen
  - Generating lattices, by using `TLG.fst` and the outputs of the network (`decode.{}.ark`). Lattices  are saved as `lat.gz` file in `exp/demo/decode_${dataset}_$lmtype`.

- **score.sh** from Eesen
  - **lattice-scale**: Scale the lattice with different acoustic scales.                                                                                                                                                                                   
  - **lattice-best-path**: Find the best path in the generated lattice.
  - **compute-wer**: Compute the WER.

**3) lmrescore_const_arpa.sh** from  Kaldi

Rescore the lattice with ConstArpa-type language model.

**4) lmrescore.sh** from Kaldi

Rescore the lattice with fst-type language model.