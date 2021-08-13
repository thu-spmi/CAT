# WSJ

* SP: 3way speed perturbation

| Model             | SP   | Eval92 | Dev93 | Param (M) | Notes                              |
| ----------------- | ---- | ------ | ----- | --------- | ---------------------------------- |
| [BLSTM](exp/demo) | Y    | 3.65   | 6.30  | 13.49     | ---                                |
| BLSTM             | N    | 3.90   | 6.24  | 13.49     | from CTC-CRF ICASSP2019            |
| BLSTM             | Y    | 3.79   | 6.23  | 13.49     | from CTC-CRF ICASSP2019            |
| BLSTM             | N    | 5.19   | 8.62  | 13.49     | char-based from CTC-CRF ICASSP2019 |
| BLSTM             | Y    | 5.32   | 8.22  | 13.49     | char-based from CTC-CRF ICASSP2019 |
| VGG-BLSTM         | Y    | 3.2    | 5.7   | 16        | ---                                |
| TDNN-NAS          | Y    | 2.77   | 5.68  | 11.9      | from ST-NAS SLT2021                |

