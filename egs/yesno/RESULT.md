# Yesno

Results on Yesno datasets.

## VGGBLSTM

* LE: Learning Rate Type.
* AM: VGGBLSTM with 9.29M parameters.
* Hyper-parameters of AM training: `lamb=0.01, n_layers=3, hdim=320, lr=0.001`

| Unit  | Test | LE                          | 
| ----- | ---- | --------------------------- | 
| phone | 2.92 | SchedulerCosineAnnealing    | 
| phone | 1.25 | SchedulerEarlyStop          | 
