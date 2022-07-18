# Yesno

Results on Yesno datasets.

## VGGBLSTM

* LE: Learning Rate Type.
* AM: VGGBLSTM with 9.29M parameters.
* Hyper-parameters of AM training: `lamb=0.01, n_layers=3, hdim=320, lr=0.001`

| Unit  | Test | LE                          | exp link                      |
| ----- | ---- | --------------------------- | ----------------------------- |
| phone | 2.92 | SchedulerCosineAnnealing    | [1-VGGBLSTM](exp/1-VGGBLSTM/) |
| phone | 1.25 | SchedulerEarlyStop          | [2-VGGBLSTM](exp/2-VGGBLSTM/) |
