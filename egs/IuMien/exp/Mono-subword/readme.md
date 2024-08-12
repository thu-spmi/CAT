### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 90.22
* GPU info \[1\]
  * \[1\] NVIDIA GeForce RTX 4090

### Notes

BPE modeling, using Mien language data for training from scratch.


### How to run exp

Please refer to the [`run.history.sh`](./run.history.sh)

### Result
We did three independent experiments, and the results of each independent experiment on its corresponding test set are as follows.

|  |  WER w/o LM| WER with LM |
|---| ---|--- |
| exp1 | 9.80 |  7.11 |
| exp2 | 10.04 | 7.04 |
| exp3 | 9.29 | 6.46 |
| avg-3 | 9.71 | 6.87 |


### training process monitor

During the training process, the loss change curve can be seen in the [training process monitor](./monitor/readme.md)
