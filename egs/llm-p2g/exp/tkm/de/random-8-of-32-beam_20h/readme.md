# Fine-tuning LLM-P2G (mT5-base) using Top-K marginalized training and decoding with 20 hours German speech data
Author: Ma, Te (mate153125@gmail.com)
### Basic info

__20 hours of `German`__ speech data and the fine-tuned model [`Whistle_ft_de_phoneme_20h`](../../../Crosslingual/de/Whistle_ft_phoneme_20h/readme.md) are used to generate phoneme sequences by beam search decoding in real time for TKM. For training, the beam width is __32__ and only __8__ phoneme sequences are randomly selected as candidates in TKM. For decoding,beam search with beam width of __8__ is used and the top-8 sequences are used in TKM.


### Training process

The script [`run_p2g.sh`](../../../../run_p2g.sh) contains the overall model training process.

#### Stage 0: Data preparation
* The data preparation has been implemented in [`Whistle_ft_phoneme_20h`](../../../Crosslingual/de/Whistle_ft_phoneme_20h/readme.md). After beam search decoding for the S2P model, we run the script [`read_nbest.py`](../../../../local/read_nbest.py) to obtain the noisy phoneme sequence from decoding results.
* The detailed model parameters are in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). The generated phoneme data paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets.

#### Stage 1 to 3: Model training

* To train tokenizer:

    The tokenizer has been trained in [`Whistle_ft_phoneme_20h`](../../../Crosslingual/de/Whistle_ft_phoneme_20h/readme.md).

* To fine-tune the mT5-base model:

        `bash run.sh de exp/tkm/de/random-8-of-32-beam_20h --sta 2 --sto 3`
* To plot the training curves:

        `python utils/plot_tb.py exp/tkm/de/random-8-of-32-beam_20h/log/tensorboard/file -o exp/tkm/de/random-8-of-32-beam_20h/monitor.png`

|     Monitor figure    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: AED decoding
* To decode with AED and calculate the %WER:

        `bash run.sh de exp/tkm/de/random-8-of-32-beam_20h --sta 4 --sto 4`

    ##### %WER
    ```
    test_de %SER 81.09 | %WER 29.20 [ 43313 / 148339, 5243 ins, 4400 del, 33670 sub ]
    ```

#### Stage 5: LM rescoring

* We use the 4-gram LM trained in [`Whistle_ft_phoneme_20h`](../../../Crosslingual/de/Whistle_ft_phoneme_20h/readme.md) to rescore the decoding results.
To rescoring and calculate the %WER:

        `bash run.sh de exp/tkm/de/random-8-of-32-beam_20h --sta 5`

    ##### %WER with 4-gram LM
    ```
    test_de %SER 80.64 | %WER 28.78 [ 42689 / 148339, 3938 ins, 5730 del, 33021 sub ]
    ```

### Resources
* The files used or generated in this experiment are available in the following table.

     | Tokenizer | Checkpoint model | Language model | Tensorboard log |
    | ----------- | ----------- | ----------- | ----------- |
    |  [`tokenizer_phn_de.tknz`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/de/tokenizer_phn_de.tknz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=1780655530&Signature=sZpxg5fqgb7x7mBiO41eASYDm1A%3D) | [`best-3.pt`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/llm-p2g/exp/de/random-8-of-32-beam_20h_best-3.pt?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064492119&Signature=mdvbnTXTFGh00v2BD2Y6EpaqGFc%3D) | [`lm_de_20h.arpa`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/de/lm_de_20h_4gram.arpa?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064482337&Signature=0AyctdR9FPc7EEz6fod3JaQ4gc0%3D) | [`tb_log`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/llm-p2g/exp/de/tb_log_random-8-of-32-beam_20h.tar.gz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064492144&Signature=pPYqasE2WRjhjnnZiTp6uCSi6%2F8%3D) |
