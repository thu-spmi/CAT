# Fine-tuning LLM-P2G (mT5-base) using data augmentation with 130 hours German noisy phoneme data
Author: Ma, Te (mate153125@gmail.com)
### Basic info

__130 hours of `German`__ noisy phoneme data was generated from the fine-tuned model [`Whistle_ft_de_phoneme_130h`](../../../Crosslingual/de/Whistle_ft_phoneme_130h/readme.md) by beam search decoding with a beam width of __32__. 


### Training process

The script [`run_p2g.sh`](../../../../run_p2g.sh) contains the overall model training process.

#### Stage 0: Data preparation
* The data preparation has been implemented in [`Whistle_ft_phoneme_130h`](../../../Crosslingual/de/Whistle_ft_phoneme_130h/readme.md). After beam search decoding for the S2P model, we run the script [`read_nbest.py`](../../../../local/read_nbest.py) to obtain the noisy phoneme sequence from decoding results.
* The detailed model parameters are in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). The generated phoneme data paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets.

#### Stage 1 to 3: Model training


* To train tokenizer:

   The tokenizer has been trained in [`Whistle_ft_phoneme_130h`](../../../Crosslingual/de/Whistle_ft_phoneme_130h/readme.md).

* To fine-tune the mT5-base model:

      `bash run.sh de exp/danp/de/32-beam --sta 2 --sto 3`
* To plot the training curves:

      `python utils/plot_tb.py exp/danp/de/32-beam/log/tensorboard/file -o exp/danp/de/32-beam/monitor.png`

|    Monitor figure   |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: AED decoding
* To decode with AED and calculate the %WER:

      `bash run.sh de exp/danp/de/32-beam --sta 4 --sto 4`

   ##### %WER
   ```
   test_de_phn_mul09   %SER 55.39 | %WER 14.17 [ 21020 / 148339, 2221 ins, 2361 del, 16438 sub ]
   ```

#### Stage 5: LM rescoring

* We use the 4-gram LM trained in [`Whistle_ft_phoneme_130h`](../../../Crosslingual/de/Whistle_ft_phoneme_130h/readme.md) to rescore the decoding results.
To rescoring and calculate the %WER:

      `bash run.sh de exp/danp/de/32-beam --sta 5`

   ##### %WER with 4-gram LM
   ```
   test_de %SER 54.70 | %WER 14.04 [ 20824 / 148339, 1476 ins, 3429 del, 15919 sub ]
   ```

### Resources
* The files used or generated in this experiment are available in the following table.

    | Tokenizer | Checkpoint model | Language model | Tensorboard log |
   | ----------- | ----------- | ----------- | ----------- |
   |  [`tokenizer_phn_de.tknz`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/de/tokenizer_phn_de.tknz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=1780655530&Signature=sZpxg5fqgb7x7mBiO41eASYDm1A%3D) | [`best-3.pt`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/llm-p2g/exp/de/32-beam_best-3.pt?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064489875&Signature=LZ%2BIlV09ioW2XAXppB5Ov5%2Fd7Eg%3D) | [`lm_de_130h.arpa`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/de/lm_de_130h_4gram.arpa?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064482365&Signature=d9O7zLIJ1mGmhoXSYo9Vd0i1UDQ%3D) | [`tb_log`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/llm-p2g/exp/de/tb_log_32-beam.tar.gz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064489917&Signature=4Z3%2Bo8kGuSMJPTd7wBklqln9fSs%3D) |


