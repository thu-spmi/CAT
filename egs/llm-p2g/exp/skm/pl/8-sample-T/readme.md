# Fine-tuning LLM-P2G (mT5-base) using Top-K marginalized training and decoding with 130 hours Polish speech data
Author: Ma, Te (mate153125@gmail.com)
### Basic info

__130 hours of `Polish`__ speech data and the fine-tuned model [`Whistle_ft_pl_phoneme_130h`](../../../Crosslingual/pl/Whistle_ft_phoneme_130h/readme.md) are used to generate phoneme sequences by sampling in real time for TKM. For training, the sample size is __8__ and all phoneme sequences are used in TKM. For decoding, beam search with beam width of __8__ is used and the top-8 sequences are used in TKM.


### Training process

The script [`run_p2g.sh`](../../../../run_p2g.sh) contains the overall model training process.

#### Stage 0: Data preparation
* The data preparation has been implemented in [`Whistle_ft_phoneme_130h`](../../../Crosslingual/pl/Whistle_ft_phoneme_130h/readme.md). 
* The detailed model parameters are in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). The generated phoneme data paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets.

#### Stage 1 to 3: Model training

* To train tokenizer:

    The tokenizer has been trained in [`Whistle_ft_phoneme_130h`](../../../Crosslingual/pl/Whistle_ft_phoneme_130h/readme.md).

* To fine-tune the mT5-base model:

        `bash run.sh pl exp/skm/pl/random-8-sample-T --sta 2 --sto 3`
* To plot the training curves:

        `python utils/plot_tb.py exp/skm/pl/random-8-sample-T/log/tensorboard/file -o exp/skm/pl/random-8-sample-T/monitor.png`

|     Monitor figure    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: AED decoding
* To decode with AED and calculate the %WER:

        `bash run.sh pl exp/skm/pl/random-8-sample-T --sta 4 --sto 4`

    ##### %WER
    ```
    test_pl %SER 13.21 | %WER 3.98 [ 2365 / 59464, 215 ins, 357 del, 1793 sub ]
    ```

#### Stage 5: LM rescoring

* We use the 4-gram LM trained in [`Whistle_ft_phoneme_130h`](../../../Crosslingual/pl/Whistle_ft_phoneme_130h/readme.md) to rescore the decoding results.
To rescoring and calculate the %WER:

        `bash run.sh pl exp/skm/pl/random-8-sample-T --sta 5`

    ##### %WER with 4-gram LM
    ```
    test_pl %SER 11.28 | %WER 3.61 [ 2149 / 59464, 138 ins, 415 del, 1596 sub ]
    ```

### Resources
* The files used or generated in this experiment are available in the following table.

     | Tokenizer | Checkpoint model | Language model | Tensorboard log |
    | ----------- | ----------- | ----------- | ----------- |
    | [`tokenizer_phn_pl.tknz`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/tokenizer_phn_pl.tknz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064482941&Signature=6E0P6xis%2FBTZjIkbdIaLS%2F%2Br%2FyU%3D) | [`best-3.pt`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/llm-p2g/exp/pl/random-8-sample-T_best-3.pt?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064495952&Signature=NXF%2FHCZMxEbnOOyk5mOj%2Bv7I4KI%3D) | [`lm_pl_130h.arpa`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/lm_pl_130h_4gram.arpa?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064492786&Signature=R2c0spDVXOPoMSpaC35EvV9Nt7k%3D) | [`tb_log`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/llm-p2g/exp/pl/tb_log_random-8-sample-T.tar.gz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064495914&Signature=anhSoy4ZxlKYSsLzCzL8QXPMQpY%3D) |

