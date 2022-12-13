# Commonvoice

Results on [Mozilla Common Voice](https://commonvoice.mozilla.org/zh-CN) dataset.

## Linguistic Resources

| resources | mul             | de                 | es               | fr              | it           |
| --------- | --------------- | ------------------ | ---------------- | --------------- | ------------ |
| lexicon   | [Mul_lexion](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/mul_lexicon/lexicon_mul.txt) | [German_lexion](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/mul_lexicon/lexicon_de.txt) | [Spanish_lexion](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/mul_lexicon/lexicon_es.txt) | [French_lexion](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/mul_lexicon/lexicon_fr.txt) | [Italian_lexion](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/mul_lexicon/lexicon_it.txt) | 
| pv        | [Mul_pv](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/mul.npy) | [German_pv](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/de.npy) | [Spanish_pv](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/es.npy) | [French_pv](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/fr.npy) | [Italian_pv](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/it.npy) |
| xlsx      | [Mul_xlsx](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/mul(de%2Cfr%2Ces%2Cit).xlsx) | [German_xlsx](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/de.xlsx) | [Spanish_xlsx](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/es.xlsx) | [French_xlsx](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/fr.xlsx) | [Italian_xlsx](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/it.xlsx) |

## Pretrained Models

| Type                                                          | Checkpoint Model                                                                                             |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [Flat-phone](exp/mc_flatphone/)                               | [VGGBLSTM](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/mul/Flat-phone/bestckpt.pt)           |
| [JoinAP-Linear](exp/mc_linear/)                               | [VGGBLSTM](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/mul/JoinAP-Linear/bestckpt.pt)        |
| [JoinAP-Nonlinear](exp/mc_nonlinear/)                         | [VGGBLSTM](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/mul/JoinAP-Nonlinear/bestckpt.pt)     |
| [JoinAP-Linear_Conformer](exp/mc_linear_conformer/)           | [Conformer](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/mul/Conformer_linear/bestckpt.pt)    |
| [JoinAP-Nonlinear_Conformer](exp/mc_nonlinear_conformer/)     | [Conformer](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/liziwei/JoinAP/mul/Coformer_nonlinear/bestckpt.pt)  |

## German Mono-lingual

### Conformer+Transformer rescoring

* Based on Mozilla Common Voice 5.1 with validated 692-hour speech and paired text.
* The `Test` column is reported in ["Advancing CTC-CRF Based End-to-End Speech Recognition with Wordpieces and Conformers"](https://arxiv.org/abs/2107.03007). <u>The Note column records the results from later, possibly crude, experiments, which may differ from the `test` column.</u>
* AM: Conformer with 25M parameters. SpecAug and 3-way perturbation is applied.
* "Trans." in the table denotes the interpolation between 4-gram and Transformer LM.
* Data for phone-based system and wp-based system rescoring respectively is publicly available on [Google Drive](https://drive.google.com/file/d/1u4C25P21ZdhytgiZbBSsO-4XSg49QIeO/view?usp=sharing), including `data/lang_{phn,bpe}`, `Nbest list`. 

| Unit                     | LM     | `Test` | Note                        | exp link |
| ------------------------ | ------ | ---- | ---------------------------- | --- |
| char  | 4-gram | 12.7 | 14.34 in exp                        | [cv_de_char](exp/cv_de_char) |
| char                     | Trans. | 11.6 | N-best with N=20, weight=0.8 | |
| phone | 4-gram | 10.7 | 10.71 in exp                        | [cv_de_phone](exp/cv_de_phone) |
| phone                    | Trans. | 10.0 | N-best with N=60, weight=0.8 | |
| wp       | 4-gram | 10.5 | 10.49 in exp                         | [cv_de_wp](exp/cv_de_wp) |
| wp                       | Trans. | 9.8  | N-best with N=20, weight=0.8 | |

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

## Multi/Cross-lingual JoinAP

<u>The exp directories contain the results from later, possibly crude, experiments, which may differ from [the JoinAP paper](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ASRU21_JoinAP.pdf).</u>

### Flat-phone

* Reported in [Chengrui Zhu, Keyu An, Huahuan Zheng and Zhijian Ou, "Multilingual and crosslingual speech recognition using phonological-vector based phone embeddings", IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), 2021.](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ASRU21_JoinAP.pdf)
* AM: VGGBLSTM with 69M parameters.3-way perturbation is applied
* Hyper-parameters of AM training: `lamb=0.01, hdim=1024, lr=0.001`

|language|w/o finetune|w/ finetune| exp link |
|---|---|---| --- |
|de|14.36|12.42| [mc_flatphone](exp/mc_flatphone) |
|fr|22.73|18.91| [mc_flatphone](exp/mc_flatphone) |
|it|25.97|21.77| [mc_flatphone](exp/mc_flatphone) |
|es|13.93|13.06| [mc_flatphone](exp/mc_flatphone) |
|pl|33.15|8.70 (10min)| [mc_flatphone](exp/mc_flatphone) |
|zh|97.10|25.39 (1h)| [mc_flatphone](exp/mc_flatphone) |

### JoinAP-Linear

* AM: VGGBLSTM_JoinAP_Linear with 69M parameters.3-way perturbation is applied
* Hyper-parameters of AM training: `lamb=0.01, hdim=1024, lr=0.001`

|language|w/o finetune| w/ finetune| exp link |
|---|---|---| --- |
|de|13.72|12.45|[mc_linear](exp/mc_linear) |
|fr|22.73|19.54|[mc_linear](exp/mc_linear) |
|it|25.85|21.70|[mc_linear](exp/mc_linear) |
|es|13.93|13.19|[mc_linear](exp/mc_linear) |
|pl|35.73 |7.50 (10min)|[mc_linear](exp/mc_linear) |
|zh|89.51 |25.21 (1h)|[mc_linear](exp/mc_linear) |

### JoinAP-Nonlinear

* AM: VGGBLSTM_JoinAP_NonLinear with 70M parameters.3-way perturbation is applied
* Hyper-parameters of AM training: `lamb=0.01, hdim=1024, lr=0.001`

|language|w/o finetune| w/ finetune| exp link |
|---|---|---| --- |
|de|13.97|12.64| [mc_nonlinear](exp/mc_nonlinear) |
|fr|22.88|19.62|[mc_nonlinear](exp/mc_nonlinear) |
|it|24.06|20.29|[mc_nonlinear](exp/mc_nonlinear) |
|es|14.10|13.26|[mc_nonlinear](exp/mc_nonlinear) |
|pl|31.80 |8.10 (10min)|[mc_nonlinear](exp/mc_nonlinear) |
|zh|88.41 |24.86 (1h)|[mc_nonlinear](exp/mc_nonlinear) |

### JoinAP-Linear_Conformer

* AM: ConformerNet_JoinAP_Linear with 47M parameters.SpecAug is applied.
* Hyper-parameters of AM training: `lamb=0.01, hdim=360, lr=0.0003`

|language|w/ finetune | exp link          |
|---|-------------------------------| -------- |
|de|11.54| [mc_linear_conformer](exp/mc_linear_conformer/) |
|fr|20.48| [mc_linear_conformer](exp/mc_linear_conformer/) |
|it|19.56| [mc_linear_conformer](exp/mc_linear_conformer/) |
|es|11.79| [mc_linear_conformer](exp/mc_linear_conformer/) |
|pl|5.38 (1h)| [mc_linear_conformer](exp/mc_linear_conformer/) |
|ja|18.99 (10h)| [mc_linear_conformer](exp/mc_linear_conformer/) |

### JoinAP-Nonlinear_Conformer

* AM: ConformerNet_JoinAP_Linear with 47M parameters.SpecAug is applied.
* Hyper-parameters of AM training: `lamb=0.01, hdim=360, lr=0.0003`

|language|w/ finetune | exp link          |
|---|-------------------------------| -------- |
|de|11.65| [mc_nonlinear_conformer](exp/mc_nonlinear_conformer/) |
|fr|20.53| [mc_nonlinear_conformer](exp/mc_nonlinear_conformer/) |
|it|19.60| [mc_nonlinear_conformer](exp/mc_nonlinear_conformer/) |
|es|11.86| [mc_nonlinear_conformer](exp/mc_nonlinear_conformer/) |
|pl|5.39 (1h)|[mc_nonlinear_conformer](exp/mc_nonlinear_conformer/) |
|ja|20.08 (10h)| [mc_nonlinear_conformer](exp/mc_nonlinear_conformer/) |

## Mono-lingual

### VGGBLSTM 

* Based on Mozilla Common Voice 9.0 with validated speech and paired text.
* Time: Data valid time (hours)
* SP: 3-way speed perturbation
* AM: VGGBLSTM with 16M parameters.
* Hyper-parameters of AM training: `lamb=0.01, n_layers=6, hdim=320, lr=0.001`

|language|Time |Unit |SP |test| exp link |
|--------|-----|-----|---|----|----------|
| be(Belarus)     | 1020 | phone| N | 8.20 | [cv_be_phone_vggblstm](exp/cv_be_phone_vggblstm/) |
| in(Indonesia)     | 26    | phone | N | 12.18 | [cv_in_phone_vggblstm](exp/cv_in_phone_vggblstm/) |
| ja(Japanese)     | 46    | phone | N | 16.04 | [cv_ja_phone_vggblstm](exp/cv_ja_phone_vggblstm/) |
| pt(Portugal)     | 115   | phone | N | 6.40  | [cv_pt_phone_vggblstm](exp/cv_pt_phone_vggblstm/) |
| ru(Russian)     | 165   | phone | N | 6.17  | [cv_ru_phone_vggblstm](exp/cv_ru_phone_vggblstm/) |
| th(Thai)     | 145   | phone | Y | 10.92 | [cv_th_phone_vggblstm](exp/cv_th_phone_vggblstm/) |
| uk(Ukraine)     | 66    | phone | N | 15.98 | [cv_uk_phone_vggblstm](exp/cv_uk_phone_vggblstm/) |
| hk(Hongkong)     | 102   | phone | N | 16.30 | [cv_hk_phone_vggblstm](exp/cv_hk_phone_vggblstm/) |
| tw(Taiwan)     | 67    | phone | N | 17.54 | [cv_tw_phone_vggblstm](exp/cv_tw_phone_vggblstm/) |   

### Conformer

* AM: ConformerNet_JoinAP_Linear with 47M parameters.SpecAug is applied.
* Hyper-parameters of AM training: `lamb=0.01, hdim=360, lr=0.0003`

|language|Time |Unit |SP |test| exp link |
|--------|-----|-----|---|----|----------|
| ar(Arab)     | 87  | phone | N | 39.83 | [cv_ar_phone_conformer](exp/cv_ar_phone_conformer/) |

* AM: ConformerNet_JoinAP_Linear with 7M parameters.SpecAug is applied.
* Hyper-parameters of AM training: `lamb=0.01, hdim=128, lr=0.0003`

|language|Time |Unit |SP |test| exp link |
|--------|-----|-----|---|----|----------|
| ur(Urdu)     | 12  | phone | N | 49.32| [cv_ur_phone_conformer](exp/cv_ur_phone_conformer/) |