# Fine-tuning LLM-P2G (mT5-base) using data augmentation with 20 hours Polish noisy phoneme data
Author: Ma, Te (mate153125@gmail.com)
### Basic info

__20 hours of `Polish`__ noisy phoneme data was generated from the fine-tuned model [`Whistle_ft_pl_phoneme_20h`](../../../Crosslingual/pl/Whistle_ft_phoneme_20h/readme.md) by beam search decoding with a beam width of __1__. 


### Training process

The script [`run_p2g.sh`](../../../../run_p2g.sh) contains the overall model training process.

#### Stage 0: Data preparation
* The data preparation has been implemented in [`Whistle_ft_pl_phoneme_20h`](../../../Crosslingual/pl/Whistle_ft_phoneme_20h/readme.md). After beam search decoding for the S2P model, we run the script [`read_nbest.py`](../../../../local/read_nbest.py) to obtain the noisy phoneme sequence from beam searching results.
* The detailed model parameters are in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). The generated phoneme data paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets.

#### Stage 1 to 3: Model training



* To train tokenizer:

    The tokenizer has been trained in [`Whistle_ft_pl_phoneme_20h`](../../../Crosslingual/pl/Whistle_ft_phoneme_20h/readme.md).

* To fine-tune the mT5-base model:

        `bash run.sh pl exp/danp/pl/1-beam_20h --sta 2 --sto 3`
* To plot the training curves:

        `python utils/plot_tb.py exp/danp/pl/1-beam_20h/log/tensorboard/file -o exp/danp/pl/1-beam_20h/monitor.png`

|     Monitor figure    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: AED decoding
* To decode with AED and calculate the %WER:

        `bash run.sh pl exp/danp/pl/1-beam_20h/ --sta 4 --sto 4`

    ##### %WER
    ```
    test_pl_20h_phn_mul_13  %SER 56.87 | %WER 23.75 [ 14124 / 59464, 1978 ins, 1295 del, 10851 sub ]
    ```

#### Stage 5: LM rescoring

* We use the 4-gram LM trained in [`Whistle_ft_pl_phoneme_20h`](../../../Crosslingual/pl/Whistle_ft_phoneme_20h/readme.md) to rescore the decoding results.
To rescoring and calculate the %WER:

        `bash run.sh pl exp/Crosslingual/pl/1-beam_20h/ --sta 5`

    ##### %WER with 4-gram LM
    ```
    test_pl_20h_phn_mul_13 %SER 51.90 | %WER 21.56 [ 12822 / 59464, 956 ins, 2123 del, 9743 sub ]
    ```

### Resources
* The files used or generated in this experiment are available in the following table.

     | tokenizer | Checkpoint model | Language model | Tensorboard log |
    | ----------- | ----------- | ----------- | ----------- |
    | [`tokenizer_phn_pl.tknz`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/tokenizer_phn_pl.tknz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064482941&Signature=6E0P6xis%2FBTZjIkbdIaLS%2F%2Br%2FyU%3D) | [`best-3.pt`](hhttp://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/llm-p2g/exp/pl/1-beam_20h_best-3.pt?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064493262&Signature=KopFxLBFMiAfSqoIN9UDECyRvzY%3D) | [`lm_pl_20h.arpa`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/lm_pl_20h_4gram.arpa?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064483620&Signature=UKV0NI43%2FzqiAV8VFbhhaLCHde0%3D) | [`tb_log`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/llm-p2g/exp/pl/tb_log_1-beam_20h.tar.gz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064493285&Signature=ePiJmzp%2FwGqCp%2BDfW6FpeYA9b9w%3D) |

