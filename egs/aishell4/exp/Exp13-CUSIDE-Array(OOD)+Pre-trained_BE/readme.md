### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 85.79
* GPU info \[1\]
  * \[1\] NVIDIA GeForce RTX 3090

### Notes

* 

### Result
```
Streaming
test_alimeeting_raw_ori %SER 90.08 | %CER 33.32 [ 20387 / 61184, 451 ins, 12682 del, 7254 sub ]
dev_alimeeting_raw_ori  %SER 88.34 | %CER 34.98 [ 6735 / 19256, 161 ins, 4332 del, 2242 sub ]
test_raw_ori    %SER 95.74 | %CER 35.97 [ 47225 / 131298, 1563 ins, 30037 del, 15625 sub ]
test_706_array_raw_ori  %SER 100.00 | %CER 35.84 [ 362 / 1010, 3 ins, 156 del, 203 sub ]
-----------------------
Non-streaming
test_alimeeting_raw_ori %SER 64.26 | %CER 17.90 [ 10952 / 61184, 393 ins, 3574 del, 6985 sub ]
dev_alimeeting_raw_ori  %SER 65.03 | %CER 19.31 [ 3719 / 19256, 162 ins, 1407 del, 2150 sub ]
test_raw_ori    %SER 82.72 | %CER 21.76 [ 28568 / 131298, 2140 ins, 10964 del, 15464 sub ]
test_706_array_raw_ori  %SER 92.50 | %CER 24.75 [ 250 / 1010, 0 ins, 89 del, 161 sub ]


```

|     training process    |
|:-----------------------:|
|![tb-plot](./monitor.png)|
