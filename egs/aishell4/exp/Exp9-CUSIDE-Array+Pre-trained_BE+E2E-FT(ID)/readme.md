### Basic info

**This part is auto-generated, add your details in Appendix**

* \# of parameters (million): 85.79
* GPU info \[4\]
  * \[4\] NVIDIA GeForce RTX 3090

### Notes

* 

### Result
```
Streaming
test_alimeeting_raw_ori %SER 66.82 | %CER 18.79 [ 11496 / 61184, 885 ins, 2283 del, 8328 sub ]
dev_alimeeting_raw_ori  %SER 67.57 | %CER 20.22 [ 3893 / 19256, 377 ins, 759 del, 2757 sub ]
test_raw_ori    %SER 78.53 | %CER 17.47 [ 22936 / 131298, 4055 ins, 3076 del, 15805 sub ]
test_706_array_raw_ori  %SER 97.50 | %CER 27.62 [ 279 / 1010, 43 ins, 15 del, 221 sub ]
-----------------------
Non-streaming
test_alimeeting_raw_ori %SER 59.40 | %CER 14.52 [ 8884 / 61184, 635 ins, 1983 del, 6266 sub ]
dev_alimeeting_raw_ori  %SER 60.93 | %CER 15.72 [ 3027 / 19256, 289 ins, 698 del, 2040 sub ]
test_raw_ori    %SER 73.09 | %CER 14.22 [ 18668 / 131298, 3477 ins, 2706 del, 12485 sub ]
test_706_array_raw_ori  %SER 92.50 | %CER 17.92 [ 181 / 1010, 8 ins, 13 del, 160 sub ]
```

|     training process    |
|:-----------------------:|
|![tb-plot](./monitor.png)|
