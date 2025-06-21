# Monolingual phoneme-based ASR model for Dutch
### Basic info

This model is built upon `Conformer` architecture and trained using the `CTC` (Connectionist Temporal Classification) approach. The training dataset consists of __70 hours of `Dutch`__ speech data sourced from the publicly available [`Common Voice`](https://commonvoice.mozilla.org/) 11.0.

* \# of parameters (million): 89.98
* GPU info 
  * NVIDIA GeForce RTX 3090
  * \# of GPUs: 5

### Training process

The script [`run.sh`](../../../run.sh) contains the overall model training process.

#### Stage 0: Data preparation
* Follow the steps [`data_prep.md`](../../../local/data_prep.md) and run [`data_prep.sh`](../../../local/data_prep.sh) to prepare the datset and pronunciation lexicon for a given language. The second and fourth stages of `data_prep.sh` involve language-specific special processing, which are detailed in the [`lang_process.md`](../../../lang-process/nl/lang_process.md). 
* The training of this model utilized 5 NVIDIA GeForce RTX 3090 GPUs and took 4 hours. The detailed model parameters are detailed in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). Dataset paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets.

#### Stage 1 to 3: Model training
* To train the model:

        `bash run.sh nl exp/Monolingual/nl --sta 1 --sto 3`
* To plot the training curves:

        `python utils/plot_tb.py exp/Monolingual/nl/log/tensorboard/file -o exp/Monolingual/nl/monitor.png`

|     Monitor figure    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: CTC decoding
* To decode with CTC and calculate the %PER:

        `bash run.sh nl exp/Monolingual/nl --sta 4 --sto 4`

    ##### %PER
    ```
    test_nl %SER 51.64 | %PER 4.60 [ 19662 / 427013, 3560 ins, 5481 del, 10621 sub ]

    ```

#### Stage 5 to 7: FST decoding
* For FST decoding, [`config.json`](./lm/config.json) and [`hyper-p.json`](./lm/hyper-p.json) are needed to train language model. Notice the distinction between the profiles for training the ASR model and the profiles for training the language model, which have the same name but are in different directories.
* To decode with FST and calculate the %WER:

        `bash run.sh nl exp/Monolingual/nl --sta 5`

    ##### %WER
    ```
    test_nl_ac1.0_lm1.4_wip0.0.hyp  %SER 34.89 | %WER 8.84 [ 8416 / 95247, 1353 ins, 1207 del, 5856 sub ]


    ```
### Resources
* The files used to train this model and the trained model are available in the following table. 

    | Tokenizer | Checkpoint model | Language model | Tensorboard log |
    | ----------- | ----------- | ----------- | ----------- |
    | [`Tokenizer`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/tokenizer.tknz?OSSAccessKeyId=LTAI5tF9KeigLW4UoLbK9vnJ&Expires=2064644893&Signature=xJjJVcnZrHh4PfGKnt3Qf7JDIHA%3D) | [`Mono_nl_best-3.pt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/nl/Mono_nl_best-3.pt) | [`lm_nl_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/nl/lm_nl_4gram.arpa) | [`tb_nl`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/nl/tb_log_nl.tar.gz) |