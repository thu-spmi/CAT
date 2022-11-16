### Basic info

**This part is auto-generated, add your details in Appendix**

* Model size/M: 115.12
* GPU info \[10\]
  * \[10\] NVIDIA GeForce RTX 3090

### Notes

- derived from `rnnt-v1`, trained with CTC-CRF phone-based
- This experiment is originally conduct on old code base, so it's not guaranteed to precisely reproduce the results.

### Data preparation

```bash
# configure experiment path
export dir=exp/crf-v1

# 0. prepare librispeech data
bash local/data_kaldi.sh

# 1. prepare librispeech lexicon
bash local/prepare_lexicon.sh

# 2. train the tokenizer
python utils/pipeline/asr.py $dir --sto 1

# 3. prepare denominator LM
mkdir -p $dir/den_meta
## configute kaldi path accoring to yours
export KALDI_ROOT=/opt/kaldi

cat data/src/train-*/text |
    bash utils/tool/prep_den_lm.sh \
        /dev/stdin $dir/den_meta/den-lm.fst \
        -tokenizer $dir/tokenizer.tknz

# 4. set den lm path in config.json, this requires hand-craft modification
# set $dir/config.json:trainer:den_lm="exp/crf-v1/tokenizer.tknz"

# 5. train the nn model
python utils/pipeline/asr.py $dir --sta 2 --sto 3

# 6. decode with FST, TODO
```


### Result
```
best 10
%WER 2.60 [ 1413 / 54402, 122 ins, 250 del, 1041 sub ] exp/crf-v1/decode_dev_clean_fglarge/wer_11_1.0
%WER 5.93 [ 3023 / 50948, 246 ins, 535 del, 2242 sub ] exp/crf-v1/decode_dev_other_fglarge/wer_17_0.0
%WER 2.93 [ 1539 / 52576, 132 ins, 299 del, 1108 sub ] exp/crf-v1/decode_test_clean_fglarge/wer_13_0.5
%WER 6.18 [ 3237 / 52343, 277 ins, 617 del, 2343 sub ] exp/crf-v1/decode_test_other_fglarge/wer_14_1.0

last 10
%WER 2.42 [ 1319 / 54402, 115 ins, 271 del, 933 sub ] exp/crf-v1//fglarge/dev_clean/wer_15_1.0
%WER 5.33 [ 2718 / 50948, 261 ins, 424 del, 2033 sub ] exp/crf-v1//fglarge/dev_other/wer_17_0.0
%WER 2.74 [ 1442 / 52576, 137 ins, 270 del, 1035 sub ] exp/crf-v1//fglarge/test_clean/wer_13_0.5
%WER 5.53 [ 2893 / 52343, 274 ins, 500 del, 2119 sub ] exp/crf-v1//fglarge/test_other/wer_14_0.5
%WER 12.24 [ 2176 / 17783, 529 ins, 431 del, 1216 sub ] exp/crf-v1//fglarge/tlv2-dev/wer_13_1.0
%WER 11.86 [ 3262 / 27500, 558 ins, 753 del, 1951 sub ] exp/crf-v1//fglarge/tlv2-test/wer_13_1.0

rescore with lm
dev_clean   %SER 26.01 | %WER 2.06 [ 1120 / 54402, 143 ins, 82 del, 895 sub ]
dev_other   %SER 39.46 | %WER 4.52 [ 2303 / 50948, 277 ins, 181 del, 1845 sub ]
test_clean  %SER 27.56 | %WER 2.35 [ 1236 / 52576, 180 ins, 85 del, 971 sub ]
test_other  %SER 43.25 | %WER 4.84 [ 2532 / 52343, 301 ins, 187 del, 2044 sub ]
```

### Monitor figure
![monitor](./monitor.jpg)
