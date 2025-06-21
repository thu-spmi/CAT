# Fine-tuning LLM-P2G (mT5-base) using Top-K marginalized training and decoding with 130 hours German speech data
Author: Ma, Te (mate153125@gmail.com)
### Basic info

__130 hours of `German`__ speech data and the fine-tuned model [`Whistle_ft_de_phoneme_130h`](../../../Crosslingual/de/Whistle_ft_phoneme_130h/readme.md) are used to generate phoneme sequences by beam search decoding in real time for TKM. The beam width is __32__ and the top-32 phoneme sequences are candidates for training and decoding in TKM. 


### Training process

The script [`run_p2g.sh`](../../../../run_p2g.sh) contains the overall model training process.

#### Stage 0: Data preparation
* The data preparation has been implemented in [`Whistle_ft_phoneme_130h`](../../../Crosslingual/de/Whistle_ft_phoneme_130h/readme.md). After beam search decoding for the S2P model, we run the script [`read_nbest.py`](../../../../local/read_nbest.py) to obtain the noisy phoneme sequence from decoding results.
* The detailed model parameters are in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). The generated phoneme data paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets.

#### Stage 1 to 3: Model training

* To train tokenizer:

    The tokenizer has been trained in [`Whistle_ft_phoneme_130h`](../../../Crosslingual/de/Whistle_ft_phoneme_130h/readme.md).

* To fine-tune the mT5-base model:

        `bash run.sh de exp/tkm/de/top-32-beam --sta 2 --sto 3`
* To plot the training curves:

        `python utils/plot_tb.py exp/tkm/de/top-32-beam/log/tensorboard/file -o exp/tkm/de/top-32-beam/monitor.png`

|     Monitor figure    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: AED decoding
* To decode with AED and calculate the %WER:

        `bash run.sh de exp/tkm/de/top-32-beam --sta 4 --sto 4`

    ##### %WER
    ```
    test_de %SER 77.92 | %WER 21.69 [ 32175 / 148339, 7293 ins, 1374 del, 23508 sub ]
    ```

#### Stage 5: LM rescoring

* We use the 4-gram LM trained in [`Whistle_ft_phoneme_130h`](../../../Crosslingual/de/Whistle_ft_phoneme_130h/readme.md) to rescore the decoding results.
To rescoring and calculate the %WER:

        `bash run.sh de exp/tkm/de/top-32-beam --sta 5`

    ##### %WER with 4-gram LM
    ```
    test_de %SER 77.32 | %WER 21.31 [ 31618 / 148339, 6863 ins, 1537 del, 23218 sub ]
    ```

### Resources
* The files used or generated in this experiment are available in the following table.

     | Tokenizer | Checkpoint model | Language model | Tensorboard log |
    | ----------- | ----------- | ----------- | ----------- |
    |  [`tokenizer_phn_de.tknz`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/de/tokenizer_phn_de.tknz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=1780655530&Signature=sZpxg5fqgb7x7mBiO41eASYDm1A%3D) | [`best-3.pt`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/llm-p2g/exp/de/top-32-beam_best-3.pt?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064492374&Signature=KR0WniNTkryJgOCNwsf1hIZfGlI%3D) | [`lm_de_130h.arpa`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/de/lm_de_130h_4gram.arpa?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064482365&Signature=d9O7zLIJ1mGmhoXSYo9Vd0i1UDQ%3D) | [`tb_log`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/llm-p2g/exp/de/tb_log_top-32-beam.tar.gz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064492406&Signature=bcOchcDr4qlCGTHTpbPCzcgvwhQ%3D) |
