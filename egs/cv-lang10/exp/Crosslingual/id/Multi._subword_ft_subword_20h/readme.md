# Fine-tuning Multi. subword model in subword form with Indonesian 20 hours data
Author: Ma, Te (mate153125@gmail.com)
### Basic info

__20 hours of `Indonesian`__ data was used to fine-tune the pretrained __subword-based multilingual ASR model__ [`Multi._subword`](../../../Multilingual/Multi._subword/readme.md) in __subword__ form. The training dataset is sourced from the publicly available [`Common Voice`](https://commonvoice.mozilla.org/) 11.0. 


### Training process

The script [`run.sh`](../../../run.sh) contains the overall model training process.

#### Stage 0: Data preparation
* The data preparation has been implemented in [`monolingual experiments for Indonesian`](../../../Monolingual/id/Mono._phoneme_20h/readme.md). Run the script [`subset.sh`](../../../../local/tools/subset.sh) to select any hours of data randomly.
* The detailed model parameters are detailed in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). Dataset paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets.

#### Stage 1 to 3: Model training
* The training of this model utilized 2 NVIDIA GeForce RTX 3090 GPUs and took 3 hours. 
  * \# of parameters (million): 90.22
  * GPU info
      * NVIDIA GeForce RTX 3090
      * \# of GPUs: 2

* For fine-tuning experiment, the output layer of the pretrained model need to be matched to the corresponding language before fine-tuning. We train the tokenizer for Indonesian and run the script [`unpack_mulingual_param.py`](../../../../local/tools/unpack_mulingual_param.py) to implement it. Then configure the parameter `init_model` in `hyper-p.json`.

* To train tokenizer:

        `bash run.sh id exp/Crosslingual/id/Multi._subword_ft_subword_20h --sta 1 --sto 1`
* To fine-tune the model:

        `bash run.sh id exp/Crosslingual/id/Multi._subword_ft_subword_20h --sta 2 --sto 3`
* To plot the training curves:

        `python utils/plot_tb.py exp/Crosslingual/id/Multi._subword_ft_subword_20h/log/tensorboard/file -o exp/Crosslingual/id/Multi._subword_ft_subword_20h/monitor.png`

|     Monitor figure    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: CTC decoding
* To decode with CTC and calculate the %PER:

        `bash run.sh id exp/Crosslingual/id/Multi._subword_ft_subword_20h/ --sta 4 --sto 4`

    ##### %WER without LM
    ```
    test_id %SER 39.77 | %WER 13.15 [ 2852 / 21685, 366 ins, 327 del, 2159 sub ]
    ```

#### Stage 5 to 7: FST decoding
* Before FST decoding, we need to train a language model for each language, which are the same as Monolingual ASR experiment. The configuration files `config.json` and `hyper-p.json` are in the `lm` of corresponding language directory in monolingual ASR experiment. Notice the distinction between the profiles for training the ASR model and the profiles for training the language model, which have the same name but are in different directories.
* To train a language model:

        `bash run.sh id exp/Crosslingual/id/Multi._subword_ft_subword_20h/ --mode subword --sta 5 --sto 5`

* To decode with FST and calculate the %WER:

        `bash run.sh id exp/Crosslingual/id/Multi._subword_ft_subword_20h/ --mode subword --sta 6`

    ##### %WER with 4-gram LM
    ```
    test_id_ac0.8_lm1.0_wip0.0.hyp  %SER 8.98 | %WER 3.07 [ 666 / 21685, 18 ins, 266 del, 382 sub ]
    ```

### Resources
* The files used to fine-tune this model and the fine-tuned model are available in the following table.

    | Word list | Checkpoint model | Language model | Tensorboard log |
    | ----------- | ----------- | ----------- | ----------- |
    | [`wordlist_id`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/wordlist_id) | [`Multi._subword_ft_subword_20h_best-3.pt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/id/Multi._subword_ft_subword_20h_best-3.pt) | [`lm_id_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/id/lm_id_4gram.arpa) | [`tb_Multi._subword_ft_subword_20h`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/id/tb_log_Multi._subword_ft_subword_20h.tar.gz) |