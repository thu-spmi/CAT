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

test_alimeeting_raw_ori %SER 89.86 | %CER 33.76 [ 20655 / 61184, 411 ins, 13157 del, 7087 sub ]
dev_alimeeting_raw_ori  %SER 87.59 | %CER 34.46 [ 6635 / 19256, 158 ins, 4374 del, 2103 sub ]
test_raw_ori    %SER 95.21 | %CER 33.77 [ 44339 / 131298, 1599 ins, 27380 del, 15360 sub ]
test_706_array_raw_ori  %SER 97.50 | %CER 33.37 [ 337 / 1010, 3 ins, 148 del, 186 sub ]
-----------------------
Non-streaming

test_alimeeting_raw_ori %SER 63.31 | %CER 17.94 [ 10976 / 61184, 389 ins, 3703 del, 6884 sub ]
dev_alimeeting_raw_ori  %SER 64.11 | %CER 18.42 [ 3547 / 19256, 159 ins, 1388 del, 2000 sub ]
test_raw_ori    %SER 81.19 | %CER 20.27 [ 26616 / 131298, 2154 ins, 9741 del, 14721 sub ]
test_706_array_raw_ori  %SER 90.00 | %CER 22.57 [ 228 / 1010, 0 ins, 86 del, 142 sub ]
```

