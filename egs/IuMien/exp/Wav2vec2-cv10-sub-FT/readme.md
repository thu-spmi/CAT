### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 90.55
* GPU info \[1\]
  * \[1\] NVIDIA GeForce RTX 4090
  
### Notes

BPE modeling, fine-tuning with Mien language data based on the Wav2Vec2-cv10 pretrained model.
 
### How to run exp

Please refer to the [`run.history.sh`](./run.history.sh)


### Result

We did three independent experiments, and the results of each independent experiment on its corresponding test set are as follows.

|  |  WER w/o LM| WER with LM |
|---| ---|--- |
| exp1 | 3.75 | 3.16  |
| exp2 | 4.08 | 3.33 |
| exp3 | 3.47 | 2.69 |
| avg-3 | 3.76 | 3.06 |

### training process monitor

During the training process, the loss change curve can be seen in the [training process monitor](./monitor/readme.md)
