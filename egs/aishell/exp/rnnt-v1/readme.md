### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 84.30
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090

### Appendix

* ported from rnnt-v15
* use torchaudio for feature extraction w/o CMVN

### Result

compared to baseline `rnnt-v15`

| model               | dev  | test |
| ------------------- | ---- | ---- |
| kaldi prep w/ CMVN  | 4.44 | 4.80 |
| kaldi prep w/o CMVN | 4.44 | 4.75 |
| torchaudio w/o CMVN | 4.43 | 4.76 |
| torchaudio w/ CMVN  | 4.60 | 5.03 |

```
beamwidth=16
dev     %SER 33.79 | %CER 4.43 [ 9104 / 205341, 154 ins, 250 del, 8700 sub ]
test    %SER 35.14 | %CER 4.76 [ 4989 / 104765, 68 ins, 195 del, 4726 sub ]

fusion lm-v5  (5-gram char) a=0.15 b=0.5 beamwidth=16
thaudio-dev     %SER 32.93 | %CER 4.35 [ 8930 / 205341, 147 ins, 325 del, 8458 sub ]
thaudio-test    %SER 34.43 | %CER 4.69 [ 4912 / 104765, 63 ins, 229 del, 4620 sub ]

rescore lm-v6 (3-gram word) a=0.28 b=-0.5 beamwidth=16
dev     %SER 31.75 | %CER 4.25 [ 8729 / 205341, 123 ins, 635 del, 7971 sub ]
test    %SER 32.78 | %CER 4.47 [ 4688 / 104765, 45 ins, 404 del, 4239 sub ]
```

### Monitor figure
![monitor](./monitor.png)
