# Multilingual phoneme-based ASR model for 10 languages(Large)
Author: Ma, Te (mate153125@gmail.com)
### Basic info

This model is built upon `Conformer` architecture and trained using the `CTC` (Connectionist Temporal Classification) approach. The training dataset consists of __4069 hours of `ten languages`__ speech data sourced from the publicly available [`Common Voice`](https://commonvoice.mozilla.org/) 11.0. 


### Training process

The script [`run.sh`](../../../run.sh) contains the overall model training process.

#### Stage 0: Data preparation
* Follow the steps [`data_prep.md`](../../../local/data_prep.md) and run [`data_prep.sh`](../../../local/data_prep.sh) to prepare the datset and pronunciation lexicon for the ten languages as following command. The second and fourth stages of `data_prep.sh` involve language-specific special processing, which are detailed in the [`lang_process.md`](../../../lang-process/) of each language directory.

        `bash run.sh ten exp/Multilingual/Multi._phoneme_L --sta 0 --sto 0`

* For phoneme-based model, a large pronunciation lexicon `lexicon_ten.txt` that contains all words in the multilingual training datasets is needed. Some words may be repeated in the pronunciation lexicon but have different pronunciations because they come from different languages, so we need to add special symbols to distinguish these words.  
* The detailed model parameters are detailed in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). Dataset paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets. For large training dataset, we should compress this data using the script [`prep_ld.py`](../../../local/tools/prep_ld.py) and configure the parameter `ld` in `hyper-p.json`.

#### Stage 1 to 3: Model training
* The training of this model utilized 10 NVIDIA GeForce RTX 3090 GPUs and took 80 hours.
  * \# of parameters (million): 89.98
  * GPU info
    * NVIDIA GeForce RTX 3090
    * \# of GPUs: 10
  
* To train the model:

        `bash run.sh ten exp/Multilingual/Multi._phoneme_L --sta 1 --sto 3`
* To plot the training curves:

        `python utils/plot_tb.py exp/Multilingual/Multi._phoneme_L/log/tensorboard/file -o exp/Multilingual/Multi._phoneme_L/monitor.png`

|     Monitor figure    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: CTC decoding
* To decode with CTC and calculate the %PER:

        `bash run.sh ten exp/Multilingual/Multi._phoneme_L --sta 4 --sto 4`

    ##### %PER
    ```
    test_en_mul     %SER 44.81 | %PER 5.42 [ 34639 / 639098, 6278 ins, 9636 del, 18725 sub ]
    test_es_mul     %SER 37.64 | %PER 1.96 [ 15113 / 769569, 3237 ins, 5049 del, 6827 sub ]
    test_fr_mul     %SER 46.90 | %PER 3.52 [ 20560 / 583591, 4081 ins, 5143 del, 11336 sub ]
    test_it_mul     %SER 44.84 | %PER 2.25 [ 17063 / 758465, 3658 ins, 5731 del, 7674 sub ]
    test_ky_mul     %SER 50.77 | %PER 4.06 [ 2635 / 64972, 424 ins, 956 del, 1255 sub ]
    test_ru_mul     %SER 50.74 | %PER 2.97 [ 14850 / 500586, 2557 ins, 5279 del, 7014 sub ]
    test_nl_mul     %SER 41.51 | %PER 2.64 [ 11270 / 427013, 2065 ins, 3120 del, 6085 sub ]
    test_tt_mul     %SER 55.95 | %PER 5.97 [ 9867 / 165304, 1349 ins, 2436 del, 6082 sub ]
    test_tr_mul     %SER 41.41 | %PER 4.04 [ 12131 / 300390, 1499 ins, 3095 del, 7537 sub ]
    test_sv_mul     %SER 72.71 | %PER 11.33 [ 18750 / 165455, 2184 ins, 4362 del, 12204 sub ]
    ```

#### Stage 5 to 7: FST decoding
* Before FST decoding, we need to train a language model for each language, which are the same as Monolingual ASR experiment. The configuration files `config.json` and `hyper-p.json` are in the corresponding language directory in monolingual ASR experiment. Notice the distinction between the profiles for training the ASR model and the profiles for training the language model, which have the same name but are in different directories.
* To train a language model for `cv-lang10`:

        `bash run.sh ten exp/Multilingual/Multi._phoneme_L --sta 5 --sto 5`

* To decode with FST and calculate the %WER for `cv-lang10`:

        `bash run.sh ten exp/Multilingual/Multi._phoneme_L --sta 6`

    ##### %WER with 4-gram LM
    ```
    test_en_mul_ac1.0_lm1.0_wip0.0.hyp      %SER 36.62 | %WER 8.80 [ 13523 / 153739, 1367 ins, 2317 del, 9839 sub ]
    test_es_mul_ac0.9_lm0.8_wip0.0.hyp      %SER 37.02 | %WER 7.02 [ 10809 / 153870, 1417 ins, 1530 del, 7862 sub ]
    test_fr_mul_ac1.0_lm1.0_wip0.0.hyp      %SER 54.60 | %WER 14.02 [ 21788 / 155399, 2322 ins, 2244 del, 17222 sub ]
    test_it_mul_ac0.9_lm1.0_wip0.0.hyp      %SER 39.39 | %WER 8.16 [ 12013 / 147151, 1894 ins, 1961 del, 8158 sub ]
    test_ky_mul_ac1.0_lm3.9_wip0.0.hyp      %SER 1.61 | %WER 0.94 [ 102 / 10889, 0 ins, 69 del, 33 sub ]
    test_nl_mul_ac0.8_lm1.0_wip0.0.hyp      %SER 27.66 | %WER 6.22 [ 5922 / 95247, 1340 ins, 623 del, 3959 sub ]
    test_ru_mul_ac1.0_lm2.4_wip0.0.hyp      %SER 5.10 | %WER 1.46 [ 1187 / 81054, 77 ins, 439 del, 671 sub ]
    test_sv_mul_ac1.0_lm2.4_wip0.0.hyp      %SER 11.86 | %WER 5.06 [ 1878 / 37126, 187 ins, 549 del, 1142 sub ]
    test_tr_mul_ac1.0_lm1.7_wip0.0.hyp      %SER 16.48 | %WER 7.05 [ 3636 / 51578, 523 ins, 323 del, 2790 sub ]
    test_tt_mul_ac1.0_lm2.0_wip0.0.hyp      %SER 18.25 | %WER 6.92 [ 1955 / 28256, 279 ins, 180 del, 1496 sub ]
    ```

### Resources
* The files used to train this model and the trained model are available in the following table. The language models in the following table are the same as Monolingual ASR experiment.

    | Pronunciation lexicon | Checkpoint model | Language model | Tensorboard log |
    | ----------- | ----------- | ----------- | ----------- |
    | [`lexicon_mul10.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/Multi._phoneme/lexicon_mul10.txt) | [`Multi._phoneme_L_best-3.pt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/Multi._phoneme_L/Multi._phoneme_L_best-3.pt) | [`lm_en_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/en/lm_en_4gram.arpa) [`lm_es_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/es/lm_es_4gram.arpa) [`lm_fr_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/fr/lm_fr_4gram.arpa) [`lm_it_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/it/lm_it_4gram.arpa) [`lm_ky_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/ky/lm_ky_4gram.arpa) [`lm_nl_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/nl/lm_nl_4gram.arpa) [`lm_ru_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/ru/lm_ru_4gram.arpa) [`lm_sv-SE_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/sv-SE/lm_sv-SE_4gram.arpa) [`lm_tr_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/tr/lm_tr_4gram.arpa) [`lm_tt_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/tt/lm_tt_4gram.arpa) | [`tb_Multi._phoneme_L`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/Multi._phoneme_L/tb_Multi._phoneme_L.tar.gz) |