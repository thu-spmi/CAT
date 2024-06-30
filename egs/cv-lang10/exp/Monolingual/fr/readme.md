# Monolingual phoneme-based ASR model for French
### Basic info

This model is built upon `Conformer` architecture and trained using the `CTC` (Connectionist Temporal Classification) approach. The training dataset consists of __823 hours of `French`__ speech data sourced from the publicly available [`Common Voice`](https://commonvoice.mozilla.org/) 11.0.

* \# of parameters (million): 89.98
* GPU info
  * NVIDIA GeForce RTX 3090
  * \# of GPUs: 8

### Training process

The script [`run.sh`](../../../run.sh) contains the overall model training process.

#### Stage 0: Data preparation
* Follow the steps [`data_prep.md`](../../../local/data_prep.md) and run [`data_prep.sh`](../../../local/data_prep.sh) to prepare the datset and pronunciation lexicon for a given language. The second and fourth stages of `data_prep.sh` involve language-specific special processing, which are detailed in the [`lang_process.md`](../../../lang-process/fr/lang_process.md). 
* The training of this model utilized 8 NVIDIA GeForce RTX 3090 GPUs and took 21 hours. The detailed model parameters are detailed in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). Dataset paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets.

#### Stage 1 to 3: Model training
* To train the model:

        `bash run.sh fr exp/Monolingual/fr --sta 1 --sto 3`
* To plot the training curves:

        `python utils/plot_tb.py exp/Monolingual/fr/log/tensorboard/file -o exp/Monolingual/fr/monitor.png`

|     Monitor figure    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: CTC decoding
* To decode with CTC and calculate the %PER:

        `bash run.sh fr exp/Monolingual/fr --sta 4 --sto 4`

    ##### %PER
    ```
    test_fr  %SER 56.81 | %PER 4.93 [ 28785 / 583591, 4987 ins, 7890 del, 15908 sub ]
    ```

#### Stage 5 to 7: FST decoding
* For FST decoding, [`config.json`](./lm/config.json) and [`hyper-p.json`](./lm/hyper-p.json) are needed to train language model. Notice the distinction between the profiles for training the ASR model and the profiles for training the language model, which have the same name but are in different directories.
* To decode with FST and calculate the %WER:

        `bash run.sh fr exp/Monolingual/fr --sta 5`

    ##### %WER
    ```
    test_fr_ac0.9_lm0.8_wip0.0.hyp  %SER 57.37 | %WER 15.58 [ 24216 / 155399, 2273 ins, 3021 del, 18922 sub ]
    ```
### Resources
* The files used to train this model and the trained model are available in the following table. 

    | Pronunciation lexicon | Checkpoint model | Language model | Tensorboard log |
    | ----------- | ----------- | ----------- | ----------- |
    | [`lexicon_fr.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/lexicon_fr.txt) | [`Mono_fr_best-3.pt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/fr/Mono_fr_best-3.pt) | [`lm_fr_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/fr/lm_fr_4gram.arpa) | [`tb_fr`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/fr/tb_log_fr.tar.gz) |