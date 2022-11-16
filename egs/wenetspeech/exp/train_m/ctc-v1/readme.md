### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 86.01
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090

### Notes

- CTC topo of `rnnt-v1`

### Result
```
no lm
dev             %SER 76.34 | %CER 12.24 [40467 / 330498, 1272 ins, 15956 del, 23239 sub ]
test_net        %SER 74.35 | %CER 15.44 [64172 / 415746, 2045 ins, 14216 del, 47911 sub ]
test_meeting    %SER 95.05 | %CER 24.04 [52979 / 220385, 1346 ins, 21329 del, 30304 sub ]
aishell-1 test  %SER 59.98 | %CER 8.78 [9202 / 104765, 339 ins, 187 del, 8676 sub ]
```

### Monitor figure
![monitor](./monitor.png)
