# Thchs30

Results on Thchs30 datasets.

## BLSTM

* SP: 3-way speed perturbation
* AM: BLSTM with 6M parameters.
* Hyper-parameters of AM training: `lamb=0.01, n_layers=3, hdim=320, lr=0.001`

| Unit  | Test | SP   | exp link                                |
| ----- | ---- | ---- | ----------------------------------------|
| phone | 7.87 | N    | [1-BSTM](exp/1-BLSTM/)                  |
| phone | 6.91 | Y    | [2-BLSTM-SpecAug](exp/2-BLSTM-SpecAug/) |


## VGG-BLSTM

* AM: VGG-BLSTM with 9M parameters. 
* Hyper-parameters of AM training: `lamb=0.01, n_layers=3, hdim=320, lr=0.001`

| Unit  | Test | SP   | exp link                              |
| ----- | ---- | ---- | ------------------------------------- |
| phone | 6.01 | Y    | [3-VGGBLSTM](exp/3-VGGBLSTM-SpecAug/) |


## Conformer

* AM: Conformer with 7M parameters. SpecAug and 3-way perturbation is applied.

| Unit  | Test | SP   | Note             | exp link                                        |
| ----- | ---- | ---- | ---------------- | ----------------------------------------------- |
| phone | 6.93 | Y    | run_conformer.sh | [4-Conformer-SpecAug](exp/4-Conformer-SpecAug/) |      
