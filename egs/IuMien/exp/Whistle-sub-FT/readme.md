### Basic info

* \# of parameters (million): 90.22
* GPU info \[1\]
  * \[1\] NVIDIA GeForce RTX 4090

### Notes

BPE modeling, fine-tuning with Mien language data based on the Whistle-small pretrained model.

### How to run exp

Please refer to the [`run.history.sh`](./run.history.sh)

### Result

We did three independent experiments, and the results of each independent experiment on its corresponding test set are as follows.

|  |  WER w/o LM| WER with LM |
|---| ---|--- |
| exp1 | 3.17 | 2.88  |
| exp2 | 3.71 | 3.29 |
| exp3 | 3.04 | 2.70 |
| avg-3 | 3.30 | 2.95 |

### training process monitor

During the training process, the loss change curve can be seen in the [training process monitor](./monitor/readme.md)
