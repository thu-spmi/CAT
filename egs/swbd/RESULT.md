# Switchboard

Results on Switchboard datasets.

## Conformer+Transformer rescoring

* Reported in ["Advancing CTC-CRF Based End-to-End Speech Recognition with Wordpieces and Conformers"](https://arxiv.org/abs/2107.03007)
* AM: Conformer with 52M parameters. SpecAug and 3-way perturbation is applied.
* "Trans." in the table denotes the interpolation between 4-gram and Transformer LM.
* Data for phone-based system and wp-based system rescoring respectively is publicly available on [Google Drive](https://drive.google.com/file/d/12NQn7an8FAjRVLkIlwHqeOjf6FBQMLz7/view?usp=sharing), including `data/lang_{phn,bpe}`, `Nbest list`.

| Unit                    | LM     | SW   | CH   | Eval2000 | Notes                              |
| ----------------------- | ------ | ---- | ---- | -------- | ---------------------------------- |
| [phone](exp/swbd_phone) | 4-gram | 7.9  | 16.1 | 12.1     | ---                                |
| phone                   | Trans. | 6.9  | 14.5 | 10.7     | N-best rescoring, N=40, weight=0.8 |
| [wp](exp/swbd_wp)       | 4-gram | 8.7  | 16.5 | 12.7     | ---                                |
| wp                      | Trans. | 7.2  | 14.8 | 11.1     | N-best rescoring, N=60, weight=0.8 |

**Experiment**

* Phone-based system

  ```shell
  bash run.sh
  ```

* Char-based or wordpiece-based

  ```shell
  bash run_wp.sh
  ```

For rescoring with "Trans.", please refer to `local/pytorchnn/readme`.

## VGG-BLSTM

* AM: VGG-BLSTM with 16M parameters. 3-way perturbation is applied.
* Hyper-parameters of AM training: `lamb=0.1, hdim=320, lr=0.001`

| Unit  | SW   | CH   |
| ----- | ---- | ---- |
| phone | 9.9  | 19.4 |

## BLSTM

* AM: BLSTM with 13M parameters.
* Hyper-parameters of AM training: `lamb=0.1, hdim=320, lr=0.001`

| Unit  | SW   | CH   | SP   |
| ----- | ---- | ---- | ---- |
| phone | 11.0 | 21.0 | N    |
| phone | 10.3 | 19.7 | Y    |
| char  | 12.7 | 24.0 | N    |
| char  | 11.4 | 21.7 | Y    |

