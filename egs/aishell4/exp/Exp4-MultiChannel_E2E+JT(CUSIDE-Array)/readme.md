### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 25.77
* GPU info \[4\]
  * \[4\] NVIDIA GeForce RTX 3090

### Notes

* 

### Result
```
Streaming
test_alimeeting_raw_ori %SER 86.03 | %CER 41.61 [ 25459 / 61184, 1200 ins, 4241 del, 20018 sub ]
dev_alimeeting_raw_ori  %SER 87.02 | %CER 45.27 [ 8717 / 19256, 570 ins, 1420 del, 6727 sub ]
test_raw_ori    %SER 91.80 | %CER 36.68 [ 48156 / 131298, 4879 ins, 5954 del, 37323 sub ]
test_706_array_raw_ori  %SER 100.00 | %CER 73.86 [ 746 / 1010, 14 ins, 298 del, 434 sub ]
-----------------------
Non-streaming
test_alimeeting_raw_ori %SER 81.88 | %CER 36.21 [ 22153 / 61184, 1144 ins, 3347 del, 17662 sub ]
dev_alimeeting_raw_ori  %SER 84.02 | %CER 40.34 [ 7767 / 19256, 555 ins, 1107 del, 6105 sub ]
test_raw_ori    %SER 89.07 | %CER 31.21 [ 40975 / 131298, 4239 ins, 4902 del, 31834 sub ]
test_706_array_raw_ori  %SER 100.00 | %CER 66.24 [ 669 / 1010, 9 ins, 226 del, 434 sub ]
```

|     training process    |
|:-----------------------:|
|![tb-plot](./monitor.png)|
