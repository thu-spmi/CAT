# Commonvoice

Results on [Mozilla Common Voice](https://commonvoice.mozilla.org/zh-CN) dataset.

## German

### Conformer+Transformer rescoring

* Based on Mozilla Common Voice 5.1 with validated 692-hour speech and paired text.
* Reported in ["Advancing CTC-CRF Based End-to-End Speech Recognition with Wordpieces and Conformers"](https://arxiv.org/abs/2107.03007)
* AM: Conformer with 25M parameters. SpecAug and 3-way perturbation is applied.
* "Trans." in the table denotes the interpolation between 4-gram and Transformer LM.
* Data for phone-based system and wp-based system rescoring respectively is publicly available on [Google Drive](https://drive.google.com/file/d/1u4C25P21ZdhytgiZbBSsO-4XSg49QIeO/view?usp=sharing), including `data/lang_{phn,bpe}`, `Nbest list`. 

| Unit                     | LM     | Test | Notes                        |
| ------------------------ | ------ | ---- | ---------------------------- |
| [char](exp/cv_de_char/)  | 4-gram | 12.7 | ---                          |
| char                     | Trans. | 11.6 | N-best with N=20, weight=0.8 |
| [phone](exp/cv_de_phone) | 4-gram | 10.7 | ---                          |
| phone                    | Trans. | 10.0 | N-best with N=60, weight=0.8 |
| [wp](exp/cv_de_wp)       | 4-gram | 10.5 | ---                          |
| wp                       | Trans. | 9.8  | N-best with N=20, weight=0.8 |

**Experiment**

Kaldi setup:

Since the commonvoice data are in `mp3` format, you need to modify the two files `utils/data/get_reco2dur.sh` and `utils/data/get_utt2dur.sh` by setting `read_entire_file=True`.

* Phone-based system

  ```shell
  bash run.sh
  ```

* Char-based or wordpiece-based

  ```shell
  bash run_wp.sh
  ```

  The default setup in `run_wp.sh` is for wp-based experiment. To run the char-based one, you need to modify the `bpemode=unigram` to`bpemode=char`.

For rescoring with "Trans.", please refer to `local/pytorchnn/readme`.

## Multi/Cross-lingual

### Flat-phone

* Reported in [Chengrui Zhu, Keyu An, Huahuan Zheng and Zhijian Ou, "Multilingual and crosslingual speech recognition using phonological-vector based phone embeddings", IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), 2021.](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ASRU21_JoinAP.pdf)
* AM: VGGBLSTM with 69M parameters.3-way perturbation is applied
* Hyper-parameters of AM training: `lamb=0.01, hdim=1024, lr=0.001`

|language|w/ finetune|w/o finetune|
|---|---|---|
|de|12.42|14.36|
|fr|18.91|22.73|
|it|21.77|25.97|
|es|13.06|13.93|
|zh|25.39 (1h)|97.10|
|pl|8.70 (10min)|33.15|

### JoinAP-Linear

* AM: VGGBLSTM_JoinAP_Linear with 69M parameters.3-way perturbation is applied
* Hyper-parameters of AM training: `lamb=0.01, hdim=1024, lr=0.001`

|language|w/ finetune|w/o finetune| 
|---|---|---|
|de|12.45|13.72|
|fr|19.54|22.73|
|it|21.70|25.85|
|es|13.19|13.93|
|zh|25.21 (1h)| 89.51 |
|pl|7.50 (10min)| 35.73 |

### JoinAP-Nonlinear

* AM: VGGBLSTM_JoinAP_NonLinear with 70M parameters.3-way perturbation is applied
* Hyper-parameters of AM training: `lamb=0.01, hdim=1024, lr=0.001`

|language|w/ finetune|w/o finetune| 
|---|---|---|
|de|12.64|13.97|
|fr|19.62|22.88|
|it|20.29|24.06|
|es|13.26|14.10|
|zh|24.86 (1h)| 88.41 |
|pl|8.10 (10min)| 31.80 |

**Experimental resources**

1. Pronunciation dictionary from [CommonVoice_lexicon](./CommonVoice_lexicon) Note: `de, es` capital.
2. Phonemic features from [CommonVoice_pv](CommonVoice_pv).
