# Thchs30

Results on Thchs30 datasets.

## BLSTM

* SP: 3-way speed perturbation
* AM: BLSTM with 6M parameters.
* Hyper-parameters of AM training: `lamb=0.01, n_layers=3, hdim=320, lr=0.001`

| Unit  | Test | SP   |
| ----- | ---- | ---- |
| phone | 7.87 | N    |
| phone | 6.91 | Y    |


## VGG-BLSTM

* AM: VGG-BLSTM with 9M parameters. 
* Hyper-parameters of AM training: `lamb=0.01, n_layers=3, hdim=320, lr=0.001`

| Unit  | Test | SP   |
| ----- | ---- | ---- |
| phone | 6.01 | Y    |


## Conformer+Transformer rescoring

* AM: Conformer with 7M parameters. SpecAug and 3-way perturbation is applied.

| Unit  | Test | SP   | Note             |
| ----- | ---- | ---- | ---------------- |
| phone | 6.93 | Y    | run_conformer.sh |       
