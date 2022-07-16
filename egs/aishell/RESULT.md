# Aishell

Results on Aishell datasets.

## BLSTM

* SP: 3-way speed perturbation
* SA: SpecAugment
* AM: BLSTM with 13M parameters.
* Hyper-parameters of AM training: `lamb=0.01, n_layers=6, hdim=320, lr=0.001`

| Unit  | Test | SP   | SA   | exp link  |
| ----- | ---- | ---- | ---- | ---- |
| phone | 6.60 | Y    | Y    |  [2-BLSTM-SpecAug](exp/2-BLSTM-SpecAug)    |
| phone | 6.48 | Y    | N    |  [1-BLSTM](exp/1-BLSTM)    |


## VGG-BLSTM

* AM: VGG-BLSTM with 16M parameters. 
* Hyper-parameters of AM training: `lamb=0.01, n_layers=6, hdim=320, lr=0.001`

| Unit  | Test | SP   | SA   | exp link  |
| ----- | ---- | ---- | ---- | ----  |
| phone | 6.18 | Y    | N    | [4-VGGBLSTM-SpecAug](exp/4-VGGBLSTM-SpecAug)|
| phone | 6.26 | Y    | Y    | [3-VGGBLSTM](exp/3-VGGBLSTM) |


## Conformer

* AM: Conformer with 7M parameters. SpecAug and 3-way perturbation is applied.

| Unit  | Test | SP   | Note             | exp link  |
| ----- | ---- | ---- | ---------------- | ----  |
| phone | 5.65 | Y    | run_conformer.sh | [5-Conformer](exp/5-Conformer) | 



