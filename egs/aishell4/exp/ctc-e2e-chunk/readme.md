### Basic info

**This part is auto-generated, add your details in Appendix**

``` 
< 400000
==================== Stage 2 Pickle data====================
pack_data(): remove 1795 unqualified sequences.
# of frames: 2502494445 | tokens: 652320 | seqs: 33930
# of frames: 133634226 | tokens: 34632 | seqs: 1881 


 < 160000
==================== Stage 2 Pickle data====================
pack_data(): remove 4330 unqualified sequences.
# of frames: 2004997055 | tokens: 520118 | seqs:31395
# of frames: 133634226 | tokens: 34632 | seqs: 1881  
```

### Notes

* data prepare
```bash
bash local/data_multi.sh -subsets train dev test -datapath /path/to/aishell4 
bash local/audio2ark_multi.sh train dev test --res 16000
```

### Result
```
流式：
dev_raw        %SER 75.56 | %CER 20.96 [ 7017 / 33474, 764 ins, 1081 del, 5172 sub ]                        
test_raw       %SER 88.64 | %CER 30.73 [ 40350 / 131298, 4646 ins, 4370 del, 31334 sub ]    

非流式：
Time = 517.67 s | RTF = 0.78                                                                       
dev_raw        %SER 75.66 | %CER 20.22 [ 6770 / 33474, 727 ins, 996 del, 5047 sub ]
test_raw       %SER 88.51 | %CER 29.65 [ 38933 / 131298, 4629 ins, 4141 del, 30163 sub ]
```
|     training process    |
|:-----------------------:|
|![monitor](./monitor.png)|



