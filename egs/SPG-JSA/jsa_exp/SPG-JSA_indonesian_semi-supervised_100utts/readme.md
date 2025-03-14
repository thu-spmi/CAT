### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 126.03
* GPU info \[9\]
  * \[9\] NVIDIA GeForce RTX 3090

### Notes

Before starting SPG-JSA training, we need three models 分别完成初始化。

- **S2P:** fine-tune Whistle (90M) on 10minutes of polish data, [experiment directory](../../s2p_exp/)|
|P2G||
|G2P||

### Result

```
# without LM
test_id %SER 26.04 | %WER 9.04 [ 1960 / 21685, 199 ins, 267 del, 1494 sub ]

# with LM
test_id_ac1.0_lm2.0_wip0.0.hyp  %SER 9.51 | %WER 3.26 [ 707 / 21685, 27 ins, 274 del, 406 sub ]

# MLS decoding "LM_weight": 0.7
test_id %SER 7.02 | %WER 2.47 [ 535 / 21685, 22 ins, 145 del, 368 sub ]

# S2P PER
test_id %SER 98.65 | %PER 20.66 [ 23448 / 113495, 377 ins, 2785 del, 20286 sub ]
```

|     training process    |
|:-----------------------:|
|![tb-plot](./monitor.png)|
