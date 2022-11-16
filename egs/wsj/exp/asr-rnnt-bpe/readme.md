### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 21.60
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090

### Notes

```bash
# prepare data
bash local/data_kaldi.sh -use-3way-sp

# train and inference
python utils/pipeline/asr.py exp/asr-rnnt-bpe
```

* RNN-T training and Beam Search decoding

### Result
```
eval92  %SER 78.08 | %WER 13.73 [ 775 / 5643, 27 ins, 207 del, 541 sub ]
dev93   %SER 83.30 | %WER 16.30 [ 1342 / 8234, 101 ins, 332 del, 909 sub ]
```

|     training process    |
|:-----------------------:|
|![monitor](./monitor.png)|
