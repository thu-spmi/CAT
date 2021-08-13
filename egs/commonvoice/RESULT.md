# Commonvoice

Results on [Mozilla Common Voice](https://commonvoice.mozilla.org/zh-CN) dataset.

## German

### Conformer+Transformer rescoring

* Based on Mozilla Common Voice 5.1 with validated 692-hour speech and paired text.
* Reported in ["Advancing CTC-CRF Based End-to-End Speech Recognition with Wordpieces and Conformers"](https://arxiv.org/abs/2107.03007)
* AM: Conformer with 25M parameters. SpecAug and 3-way perturbation is applied.
* "Trans." in the table denotes the interpolation between 4-gram and Transformer LM.
* Data for phone-based system and wp-based system rescoring respectively is publicly available on [Google Drive](https://drive.google.com/file/d/1u4C25P21ZdhytgiZbBSsO-4XSg49QIeO/view?usp=sharing), including `data/lang_{phn,bpe}`, `Nbest list`. 

| Unit                     | LM     | Test | Notes                        |
| ------------------------ | ------ | ---- | ---------------------------- |
| [char](exp/cv_de_char/)  | 4-gram | 12.7 | ---                          |
| char                     | Trans. | 11.6 | N-best with N=20, weight=0.8 |
| [phone](exp/cv_de_phone) | 4-gram | 10.7 | ---                          |
| phone                    | Trans. | 10.0 | N-best with N=60, weight=0.8 |
| [wp](exp/cv_de_wp)       | 4-gram | 10.5 | ---                          |
| wp                       | Trans. | 9.8  | N-best with N=20, weight=0.8 |

**Experiment**

Kaldi setup:

Since the commonvoice data are in `mp3` format, you need to modify the two files `utils/data/get_reco2dur.sh` and `utils/data/get_utt2dur.sh` by setting `read_entire_file=True`.

* Phone-based system

  ```shell
  bash run.sh
  ```

* Char-based or wordpiece-based

  ```shell
  bash run_wp.sh
  ```

  The default setup in `run_wp.sh` is for wp-based experiment. To run the char-based one, you need to modify the `bpemode=unigram` to`bpemode=char`.

For rescoring with "Trans.", please refer to `local/pytorchnn/readme`.

