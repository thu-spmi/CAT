# Fine-tuning subword model in subword form with Polish 130 hours data
Author: Ma, Te (mate153125@gmail.com)
### Basic info

__130 hours of `Polish`__ data was used to fine-tune the pretrained __subword-based multilingual ASR model__ [`Multi._subword`](../../../Multilingual/Multi._subword/readme.md) in __subword__ form. The training dataset was randomly selected from 130 hours Polish dataset sourced from the publicly available [`Common Voice`](https://commonvoice.mozilla.org/) 11.0. 


### Training process

The script [`run.sh`](../../../run.sh) contains the overall model training process.

#### Stage 0: Data preparation
* The data preparation has been implemented in [`monolingual experiments for Polish`](../../../Monolingual/pl/Mono._phoneme_130h/readme.md). Run the script [`subset.sh`](../../../../local/tools/subset.sh) to select any hours of data randomly.
* The detailed model parameters are detailed in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). Dataset paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets.

#### Stage 1 to 3: Model training
* The training of this model utilized 5 NVIDIA GeForce RTX 3090 GPUs and took 18 hours. 
  * \# of parameters (million): 90.22
  * GPU info
      * NVIDIA GeForce RTX 3090
      * \# of GPUs: 5

* For fine-tuning experiment, the output layer of the pretrained model need to be matched to the corresponding language before fine-tuning. We train the tokenizer for Polish and run the script [`unpack_mulingual_param.py`](../../../../local/tools/unpack_mulingual_param.py) to implement it. Then configure the parameter `init_model` in `hyper-p.json`.

* To train tokenizer:

        `bash run.sh pl exp/Crosslingual/pl/Multi._subword_ft_subword_130h --sta 1 --sto 1`
* To fine-tune the model:

        `bash run.sh pl exp/Crosslingual/pl/Multi._subword_ft_subword_130h --sta 2 --sto 3`
* To plot the training curves:

        `python utils/plot_tb.py exp/Crosslingual/pl/Multi._subword_ft_subword_130h/log/tensorboard/file -o exp/Crosslingual/pl/Multi._subword_ft_subword_130h/monitor.png`

|     Monitor figure    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: CTC decoding
* To decode with CTC and calculate the %PER:

        `bash run.sh pl exp/Crosslingual/pl/Multi._subword_ft_subword_130h/ --sta 4 --sto 4`

    ##### %WER without LM
    ```
    test_pl %SER 17.92 | %WER 5.44 [ 3236 / 59464, 333 ins, 439 del, 2464 sub ]
    ```

#### Stage 5 to 7: FST decoding
* Before FST decoding, we need to train a language model for each language, which are the same as Monolingual ASR experiment. The configuration files `config.json` and `hyper-p.json` are in the `lm` of corresponding language directory in monolingual ASR experiment. Notice the distinction between the profiles for training the ASR model and the profiles for training the language model, which have the same name but are in different directories.
* To train a language model:

        `bash run.sh pl exp/Crosslingual/pl/Multi._subword_ft_subword_130h/ --mode subword --sta 5 --sto 5`

* To decode with FST and calculate the %WER:

        `bash run.sh pl exp/Crosslingual/pl/Multi._subword_ft_subword_130h/ --mode subword --sta 6`

    ##### %WER with 4-gram LM
    ```
    test_pl_ac1.0_lm0.8_wip0.0.hyp  %SER 11.39 | %WER 3.76 [ 2237 / 59464, 177 ins, 475 del, 1585 sub ]
    ```

### Resources
* The files used to fine-tune this model and the fine-tuned model are available in the following table.

    | Tokenizer | Checkpoint model | Language model | Tensorboard log |
    | ----------- | ----------- | ----------- | ----------- |
    | [`Tokenizer`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/pl/wordlist_pl?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064642450&Signature=fiZYZSh%2B6EL%2BS5XwMm2l%2Bm15qLo%3D) | [`Multi._subword_ft_subword_130h_best-3.pt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/pl/Multi._subword_ft_subword_130h_best-3.pt) | [`lm_pl_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/pl/lm_pl_4gram.arpa) | [`tb_Multi._subword_ft_subword_130h`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/pl/tb_log_Multi._subword_ft_subword_130h.tar.gz) |