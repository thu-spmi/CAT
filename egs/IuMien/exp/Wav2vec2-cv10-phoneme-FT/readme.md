### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 90.21
* GPU info \[1\]
  * \[1\] NVIDIA GeForce RTX 4090

### Notes

Phone modeling, fine-tuning with Mien language data based on the Wav2vec2-cv10 pretrained model.

### How to run exp

Please refer to the [`run.history.sh`](./run.history.sh)

### Result

We did three independent experiments, and the results of each independent experiment on its corresponding test set are as follows.

|  |  PER | WER  |
|---| ---|--- |
| exp1 | 2.40 | 2.71  |
| exp2 | 2.82 | 3.06 |
| exp3 | 2.39 | 2.53 |
| avg-3 | 2.53 | 2.76 |

### training process monitor

During the training process, the loss change curve can be seen in the [training process monitor](./monitor/readme.md)
