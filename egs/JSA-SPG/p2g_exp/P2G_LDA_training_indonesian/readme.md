### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 18.15
* GPU info \[4\]
  * \[4\] NVIDIA GeForce RTX 3090

### Notes

* 

### Result
```
test_id_jas_s2p        %SER 83.80 | %WER 30.69 [ 5079 / 16549, 515 ins, 847 del, 3717 sub ]

# SPG-JSA  decoding
test_indo       %SER 83.80 | %WER 30.69 [ 5079 / 16549, 515 ins, 847 del, 3717 sub ]
test_indo_ac1.0_lm4.1_wip0.0.hyp        %SER 33.67 | %WER 12.68 [ 2098 / 16549, 47 ins, 1274 del, 777 sub ]

# MLS decoding with "LM_weight": 0.6
test_indo       %SER 28.27 | %WER 11.23 [ 1858 / 16549, 49 ins, 1179 del, 630 sub ]
```

|     training process    |
|:-----------------------:|
|![tb-plot](./monitor.png)|
