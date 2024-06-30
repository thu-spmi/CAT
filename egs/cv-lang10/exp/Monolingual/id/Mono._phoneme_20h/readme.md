# Monolingual phoneme-based ASR model for Indonesian
### Basic info

This model is built upon `Conformer` architecture and trained using the `CTC` (Connectionist Temporal Classification) approach. The training dataset consists of __20 hours of `Indonesian`__ speech data sourced from the publicly available [`Common Voice`](https://commonvoice.mozilla.org/) 11.0.


### Training process

The script [`run.sh`](../../../run.sh) contains the overall model training process.

#### Stage 0: Data preparation
* Follow the steps [`data_prep.md`](../../../local/data_prep.md) and run [`data_prep.sh`](../../../local/data_prep.sh) to prepare the datset and pronunciation lexicon for a given language. The second and fourth stages of `data_prep.sh` involve language-specific special processing, which are detailed in the [`lang_process.md`](../../../lang-process/id/lang_process.md). 
* The detailed model parameters are detailed in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). Dataset paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets.

#### Stage 1 to 3: Model training
* The training of this model utilized 1 NVIDIA GeForce RTX 3090 GPUs and took 10 hours.
    * \# of parameters (million): 89.98
    * GPU info 
        * NVIDIA GeForce RTX 3090
        * \# of GPUs: 1

* To train the model:

        `bash run.sh id exp/Monolingual/id/Mono._phoneme_20h --sta 1 --sto 3`
* To plot the training curves:

        `python utils/plot_tb.py exp/Monolingual/id/Mono._phoneme_20h/log/tensorboard/file -o exp/Monolingual/id/Mono._phoneme_20h/monitor.png`

|     Monitor figure    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: CTC decoding
* To decode with CTC and calculate the %PER:

        `bash run.sh id exp/Monolingual/id/Mono._phoneme_20h --sta 4 --sto 4`

    ##### %PER
    ```
    test_id %SER 43.28 | %PER 5.74 [ 6513 / 113495, 1047 ins, 1650 del, 3816 sub ]

    ```

#### Stage 5 to 7: FST decoding
* For FST decoding, [`config.json`](./lm/config.json) and [`hyper-p.json`](./lm/hyper-p.json) are needed to train language model. Notice the distinction between the profiles for training the ASR model and the profiles for training the language model, which have the same name but are in different directories.
* To decode with FST and calculate the %WER:

        `bash run.sh id exp/Monolingual/id/Mono._phoneme_20h --sta 5`

    ##### %WER
    ```
    test_id_ac1.0_lm2.4_wip0.0.hyp  %SER 8.24 | %WER 3.28 [ 712 / 21685, 40 ins, 225 del, 447 sub ]


    ```
### Resources
* The files used to train this model and the trained model are available in the following table. 

    | Pronunciation lexicon | Checkpoint model | Language model | Tensorboard log |
    | ----------- | ----------- | ----------- | ----------- |
    | [`lexicon_id.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/lexicon_id.txt) | [`Mono._phoneme_20h_best-3.pt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/id/Mono._phoneme_20h_best-3.pt) | [`lm_id_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/id/lm_id_4gram.arpa) | [`tb_Mono._phoneme_20h_id`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/id/tb_log_Mono._phoneme_20h.tar.gz) |
