# Aishell

Results on Aishell datasets.

## BLSTM

* SP: 3-way speed perturbation
* AM: BLSTM with 13M parameters.
* Hyper-parameters of AM training: `lamb=0.01, n_layers=6, hdim=320, lr=0.001`

| Unit  | Test | SP   |
| ----- | ---- | ---- |
| phone | 6.60 | N    |
| phone | 6.48 | Y    |


## VGG-BLSTM

* AM: VGG-BLSTM with 16M parameters. 
* Hyper-parameters of AM training: `lamb=0.01, n_layers=6, hdim=320, lr=0.001`

| Unit  | Test | SP   |
| ----- | ---- | ---- |
| phone | 6.18 | N    |
| phone | 6.26 | Y    |