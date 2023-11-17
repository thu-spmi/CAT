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
dev_raw        %SER 79.27 | %CER 23.62 [ 8176 / 34613, 907 ins, 1180 del, 6089 sub ]
test_raw       %SER 90.35 | %CER 32.98 [ 43304 / 131298, 4983 ins, 4664 del, 33657 sub ]
非流式：
dev_raw        %SER 77.88 | %CER 21.13 [ 7315 / 34613, 831 ins, 1044 del, 5440 sub ]test_raw       %SER 88.51 | %CER 29.78 [ 39104 / 131298, 4404 ins, 4327 del, 30373 sub ] 
```

|     real right context   |
|:-----------------------:|
|![real right context](./right_context.png)|

|     simu right context   |
|:-----------------------:|
|![simu right context](./simu_right_context.png)|

