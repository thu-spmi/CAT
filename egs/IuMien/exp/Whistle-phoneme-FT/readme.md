### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 89.99
* GPU info \[1\]
  * \[1\] NVIDIA GeForce RTX 4090
  
### Notes

Phone modeling, fine-tuning with Mien language data based on the Whistle-small pretrained model.


### How to run exp

Please refer to the [`run.history.sh`](./run.history.sh)

### Result
We did three independent experiments, and the results of each independent experiment on its corresponding test set are as follows.

|  |  PER | WER  |
|---| ---|--- |
| exp1 | 2.45 | 2.93  |
| exp2 | 2.65 | 3.08 |
| exp3 | 2.13 | 2.38 |
| avg-3 | 2.41 | 2.71 |

### training process monitor

During the training process, the loss change curve can be seen in the [training process monitor](./monitor/readme.md)
