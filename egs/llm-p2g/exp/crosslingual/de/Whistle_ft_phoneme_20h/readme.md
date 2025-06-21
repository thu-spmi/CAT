# Fine-tuning Whistle model in phoneme form with 20 hours German data
Author: Ma, Te (mate153125@gmail.com)
### Basic info

__20 hours of `German`__ data was used to fine-tune the pretrained __phoneme-based multilingual ASR model__ [`Multi._phoneme_S`](../../../../../cv-lang10/exp/Multilingual/Multi._phoneme_S/readme.md) in __phoneme__ form. The training dataset was randomly selected from 130 hours German dataset sourced from the publicly available [`Common Voice`](https://commonvoice.mozilla.org/) 11.0. 


### Training process

The script [`run.sh`](../../../../../cv-lang10/run.sh) contains the overall model training process.

#### Stage 0: Data preparation
* Follow the steps [`data_prep.md`](../../../../../cv-lang10/local/data_prep.md) and run [`data_prep.sh`](../../../../../cv-lang10/local/data_prep.sh) to prepare the datset and word list for a given language. Then Run the script [`subset.sh`](../../../../../cv-lang10/local/tools/subset.sh) to select 20 hours of data randomly.
* The detailed model parameters are detailed in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). Dataset paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets.

#### Stage 1 to 3: Model training

* For fine-tuning experiment, the output layer of the pretrained model need to be matched to the corresponding language before fine-tuning. We train the tokenizer for German and run the script [`unpack_mulingual_param.py`](../../../../../cv-lang10/local/tools/unpack_mulingual_param.py) to implement it. Then configure the parameter `init_model` in `hyper-p.json`.

* To train tokenizer:

        `bash run.sh de exp/Crosslingual/de/Whistle_ft_phoneme_20h --sta 1 --sto 1`
* To fine-tune the model:

        `bash run.sh de exp/Crosslingual/de/Whistle_ft_phoneme_20h --sta 2 --sto 3`
* To plot the training curves:

        `python utils/plot_tb.py exp/Crosslingual/de/Whistle_ft_phoneme_20h/log/tensorboard/file -o exp/Crosslingual/de/Whistle_ft_phoneme_20h/monitor.png`

|     Monitor figure    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: CTC decoding
* To decode with CTC and calculate the %PER:

        `bash run.sh de exp/Crosslingual/de/Whistle_ft_phoneme_20h/ --sta 4 --sto 4`

    ##### %PER
    ```
    test_de %SER 88.70 | %PER 10.29 [ 82785 / 804481, 9321 ins, 16423 del, 57041 sub ]
    ```

#### Stage 5 to 7: FST decoding
* Before FST decoding, we need to train a language model with the 20 hours training data for each language. The configuration files `config.json` and `hyper-p.json` are in the `lm` of corresponding language directory in monolingual ASR experiment. Notice the distinction between the profiles for training the ASR model and the profiles for training the language model, which have the same name but are in different directories.
* To train a language model:

        `bash run.sh de exp/Crosslingual/de/Whistle_ft_phoneme_20h/ --sta 5 --sto 5`

* To decode with FST and calculate the %WER:

        `bash run.sh de exp/Crosslingual/de/Whistle_ft_phoneme_20h/ --sta 6`

    ##### %WER with 4-gram LM
    ```
    test_de_ac1.0_lm2.0_wip0.0.hyp  %SER 81.53 | %WER 30.71 [ 45554 / 148339, 9028 ins, 3796 del, 32730 sub ]
    ```

### Resources
* The files used or generated in this experiment are available in the following table.

    | Tokenizer | Checkpoint model | Language model | Tensorboard log |
    | ----------- | ----------- | ----------- | ----------- |
    | [`tokenizer_phn_de.tknz`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/de/tokenizer_phn_de.tknz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=1780655530&Signature=sZpxg5fqgb7x7mBiO41eASYDm1A%3D) | [`best-3.pt`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/de/Whistle_ft_phoneme_20h_best-3.pt?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064482597&Signature=MMXdqW2e7SkOu2q883ose4FVCsM%3D) | [`lm_de_20h.arpa`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/de/lm_de_20h_4gram.arpa?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064482337&Signature=0AyctdR9FPc7EEz6fod3JaQ4gc0%3D) | [`tb_log`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/de/tb_log_Whistle_ft_phoneme_20h.tar.gz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064482719&Signature=EReTejixETUory9TCo0DY5DUI14%3D) |