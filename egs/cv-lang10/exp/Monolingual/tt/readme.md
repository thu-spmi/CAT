# Monolingual phoneme-based ASR model for Tatar
### Basic info

This model is built upon `Conformer` architecture and trained using the `CTC` (Connectionist Temporal Classification) approach. The training dataset consists of __21 hours of `Tatar`__ speech data sourced from the publicly available [`Common Voice`](https://commonvoice.mozilla.org/) 11.0.

* \# of parameters (million): 89.98
* GPU info 
  * NVIDIA GeForce RTX 3090
  * \# of GPUs: 5

### Training process

The script [`run.sh`](../../../run.sh) contains the overall model training process.

#### Stage 0: Data preparation
* Follow the steps [`data_prep.md`](../../../local/data_prep.md) and run [`data_prep.sh`](../../../local/data_prep.sh) to prepare the datset and pronunciation lexicon for a given language. The second and fourth stages of `data_prep.sh` involve language-specific special processing, which are detailed in the [`lang_process.md`](../../../lang-process/tt/lang_process.md). 
* The training of this model utilized 5 NVIDIA GeForce RTX 3090 GPUs and took 2 hours. The detailed model parameters are detailed in [`config.json`](config.json) and [`hyper-p.json`](hyper-p.json). Dataset paths should be added to the [`metainfo.json`](../../../data/metainfo.json) for efficient management of datasets.

#### Stage 1 to 3: Model training
* To train the model:

        `bash run.sh tt exp/Monolingual/tt --sta 1 --sto 3`
* To plot the training curves:

        `python utils/plot_tb.py exp/Monolingual/tt/log/tensorboard/file -o exp/Monolingual/tt/monitor.png`

|     Monitor figure    |
|:-----------------------:|
|![tb-plot](./monitor.png)|

#### Stage 4: CTC decoding
* To decode with CTC and calculate the %PER:

        `bash run.sh tt exp/Monolingual/tt --sta 4 --sto 4`

    ##### %PER
    ```
    test_tt %SER 70.57 | %PER 10.54 [ 17428 / 165304, 2366 ins, 5034 del, 10028 sub ]
    ```

#### Stage 5 to 7: FST decoding
* For FST decoding, [`config.json`](./lm/config.json) and [`hyper-p.json`](./lm/hyper-p.json) are needed to train language model. Notice the distinction between the profiles for training the ASR model and the profiles for training the language model, which have the same name but are in different directories.
* To decode with FST and calculate the %WER:

        `bash run.sh tt exp/Monolingual/tt --sta 5`

    ##### %WER
    ```
    test_tt_ac1.0_lm2.1_wip0.0.hyp  %SER 23.79 | %WER 9.75 [ 2756 / 28256, 257 ins, 430 del, 2069 sub ]
    ```
### Resources
* The files used to train this model and the trained model are available in the following table. 

    | Pronunciation lexicon | Checkpoint model | Language model | Tensorboard log |
    | ----------- | ----------- | ----------- | ----------- |
    | [`lexicon_tt.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tt/lexicon_tt.txt) | [`Mono_tt_best-3.pt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/tt/Mono_tt_best-3.pt) | [`lm_tt_4gram.arpa`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/tt/lm_tt_4gram.arpa) | [`tb_tt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/tt/tb_log_tt.tar.gz) |