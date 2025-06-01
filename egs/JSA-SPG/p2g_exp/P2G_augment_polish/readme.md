### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 18.15
* GPU info \[5\]
  * \[5\] NVIDIA GeForce RTX 3090

### Notes

* 

### Result
```
test_pl_jsa_s2p %SER 14.79 | %WER 4.65 [ 2763 / 59464, 208 ins, 546 del, 2009 sub ]

# SPG-JSA  decoding
test_pl %SER 14.79 | %WER 4.65 [ 2763 / 59464, 208 ins, 546 del, 2009 sub ]
test_pl_ac1.0_lm1.0_wip0.0.hyp  %SER 12.88 | %WER 4.37 [ 2599 / 59464, 144 ins, 588 del, 1867 sub ]

# MLS decoding with "LM_weight": 0.7
test_pl %SER 11.04 | %WER 3.64 [ 2167 / 59464, 163 ins, 372 del, 1632 sub ]
```

|     training process    |
|:-----------------------:|
|![tb-plot](./monitor.png)|
