### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 90.22
* GPU info \[1\]
  * \[1\] NVIDIA GeForce RTX 4090
  
### Notes

BPE modeling, fine-tuning with Mien language data based on a pretrained model with subwords from cv-10.

### How to run exp

Please refer to the [`run.history.sh`](./run.history.sh)

### Result
We did three independent experiments, and the results of each independent experiment on its corresponding test set are as follows.

|  |  WER w/o LM| WER with LM |
|---| ---|--- |
| exp1 | 4.18 | 3.42  |
| exp2 | 4.79 | 3.92 |
| exp3 | 4.02 | 3.05 |
| avg-3 | 4.33 | 3.46 |

### training process monitor

During the training process, the loss change curve can be seen in the [training process monitor](./monitor/readme.md)
