# Aishell

Results on Aishell datasets.

## BLSTM

* SP: 3-way speed perturbation
* SA: SpecAugment
* AM: BLSTM with 13M parameters.
* Hyper-parameters of AM training: `lamb=0.01, n_layers=6, hdim=320, lr=0.001`

| Unit  | Test | SP   | SA   |
| ----- | ---- | ---- | ---- |
| phone | 6.60 | Y    | Y    |
| phone | 6.48 | Y    | N    |


## VGG-BLSTM

* AM: VGG-BLSTM with 16M parameters. 
* Hyper-parameters of AM training: `lamb=0.01, n_layers=6, hdim=320, lr=0.001`

| Unit  | Test | SP   | SA   |
| ----- | ---- | ---- | ---- |
| phone | 6.18 | Y    | N    |
| phone | 6.26 | Y    | Y    |


## Conformer+Transformer rescoring

* AM: Conformer with 7M parameters.3-gram language model and 3-way perturbation is applied.

| Unit  | Test | SP   | Note             |
| ----- | ---- | ---- | ---------------- |
| phone | 5.65 | Y    | run_conformer.sh |       



