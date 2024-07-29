## Data
The AISHELL-4 is a sizable real-recorded Mandarin speech dataset collected by 8-channel circular microphone array for speech processing in conference scenario, about 40 hours for non overlapping parts of the dataset

**Data prepare**

Use one of the following way:

- Prepare data with `torchaudio`: run following command to get help

   ```bash
   bash local/data_multi.sh -h
   bash local/audio2ark_multi.sh -h
   ```

Source data info will be automatically stored at `data/metainfo.json`. You can run

```bash
cd /path/to/aishell4
python utils/data/resolvedata.py
```
to refresh the information. Manually modifying is also OK.

## Result

Data prepare with command:

```bash
bash local/data_multi.sh -subsets train dev test -datapath /path/to/aishell4 
bash local/audio2ark_multi.sh train dev test --res 16000
```

Summarize experiments here.

NOTE: some of the experiments are conduct on previous code base, therefore, the settings might not be compatible to the latest. In that case, you could:

- \[Recommand\] manually modify the configuration files (`config.json` and `hyper-p.json`);

- Checkout to old code base by `hyper-p:commit` info. This could definitely reproduce the reported results, but some modules might be buggy.


### Results on AISHELL-4 test set

#### Evaluated by CER (%)  [librosa features]

| **Exp** | **Model** | **Params (M)** | **right ctx in training (ms)** | **right ctx in streaming recog. (ms)** | **Latency (ms)** | **CER** |  
| --- | --- | --- | --- | --- | --- | --- |  
| 1 | [bf+ctc](exp/ctc-e2e-chunk) | 25.77 | 400 or 0 | 0 | 400 | 34.14 (28.27) |  
| 2 | [bf+ctc+simu](exp/ctc-e2e-chunk+simu) | 27.64 | 400 or 0 or [400] | [400] | 400 + 2 | 32.18 (28.70) |  




#### Evaluated by CER (%)  [kaldi features]

Exp 2 and 4 denote joint training of chunk-based streaming and non-streaming unified model.

For Exp 1 and 3, no joint training means that whole-utterance models are trained with no chunking.

The non-streaming recognition results are in parentheses.
The right context (abbreviated as ctx) is not used by default during decoding. 

Exp 5 and 6 denote that real and simulated right contexts are used in decoding respectively.

Channel 0 is used for single-channel experiments.

$\square$: not applied.

| **Exp** | **Model** | **Params (M)** | **right ctx in training (ms)** | **right ctx in streaming recog. (ms)** | **Latency (ms)** | **CER** |  
| --- | --- | --- | --- | --- | --- | --- |  
| 1 | [Single-channel E2E](exp/Exp1-SingalChannel_E2E/readme.md) | 20.70 | $\square$ | 0 | 400 | 55.07 (38.76) |  
| 2 | [&nbsp;&nbsp;+ joint training](exp/Exp2-SingalChannel_E2E+JT(CUSIDE)/readme.md) | 20.70 | 400 or 0 | 0 | 400 | 40.95 (36.17) |  
| 3 | [Multi-channel E2E](exp/Exp3-MultiChannel_E2E/readme.md )| 25.77 | $\square$ | 0 | 400 | 56.84 (27.93) |  
| 4 |  [&nbsp;&nbsp;+ joint training](exp/Exp4-MultiChannel_E2E+JT(CUSIDE-Array)/readme.md) | 25.77 | 400 or 0 | 0 | 400 | 36.68 (31.21) |  
| 5 |  [&nbsp;&nbsp;&nbsp;&nbsp;+ real right ctx (400ms)](exp/Exp5-CUSIDE-Array+real_right_ctx/readme.md) | 25.77 | 400 or 0 | 400 | 800 | 32.51 (31.21) |  
| 6 |  [&nbsp;&nbsp;&nbsp;&nbsp;+ simu right ctx (400ms)](exp/Exp6-CUSIDE-Array+simu_right_ctx/readme.md) | 27.64 | 400 or 0 or [400] | [400] | 400 + 2 | 35.96 (31.70) |

#### In-distribution (ID) and out-of-distribution (OOD) streaming and non-streaming results (in parentheses)

ID results are underlined. 

E2E-FT refers to joint fine-tuning (FT) of the front-end (FE) and back-end (BE), with ID data (i.e. AISHELL-4) and simulated data for fine-tuning. 

Alimeeting-FE denotes the front-end from the ME2E CUSIDE-array model trained on Alimeeting. 

[MFCCA](https://www.modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary) adopts the all-neural approach of ME2E ASR, while CUSIDE-array belongs to the neural beamformer approach.


| Exp | Model | Params (M) | AISHELL-4 test | Ali-test | Ali-eval | XMOS test | Average |
|-----|-------|------------|----------------|----------|----------|-----------|---------|
| 2   | [Single-ch. E2E (CUSIDE)](exp/Exp2-SingalChannel_E2E+JT(CUSIDE)/readme.md) | 20.70 | <u>40.95 (36.17)</u> | 46.26 (41.23) | 50.10 (45.00) | 87.33 (86.34) | 56.16 (52.19) |
| 7   | &nbsp;&nbsp;[+ Pre-trained BE plug in](exp/Exp7-CUSIDE+Pre-trained_BE/readme.md) | 80.72 | <u>35.70 (26.41)</u> | 28.83 (20.29) | 29.07 (20.55) | 41.09 (29.80) | 33.67 (24.26) |
| 4   | [Multi-ch. E2E (CUSIDE-array)](exp/Exp4-MultiChannel_E2E+JT(CUSIDE-Array)/readme.md) | 25.77 | <u>36.68 (31.21)</u> | 41.61 (36.21) | 45.27 (40.34) | 73.86 (66.24) | 49.36 (43.50) |
| 8   | &nbsp;&nbsp;[+ Pre-trained BE plug in](exp/Exp8-CUSIDE-Array+Pre-trained_BE/readme.md) | 85.79 | <u>33.77 (20.27)</u> | 33.76 (17.94) | 34.46 (18.42) | 33.37 (22.57) | 33.84 (19.80) |
| 9   | &nbsp;&nbsp;&nbsp;&nbsp;[+ E2E-FT with ID (40h)](exp/Exp9-CUSIDE-Array+Pre-trained_BE+E2E-FT(ID)/readme.md) | 85.79 | <u>17.47 (14.22)</u> | 18.79 (14.52) | 20.22 (15.72) | 27.62 (17.92) | 21.03 (15.60) |
| 10  | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[+ E2E-FT with simu (13h)](exp/Exp10~12-CUSIDE-Array+Pre-trained_BE+E2E-FT(ID+simu_data)/readme.md) | 85.79 | <u>17.49 (14.14)</u> | 18.04 (13.83) | 19.11 (14.95) | 25.84 (20.69) | 20.12 (15.90) |
| 11  | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[+ E2E-FT with simu (73h)](exp/Exp10~12-CUSIDE-Array+Pre-trained_BE+E2E-FT(ID+simu_data)/readme.md) | 85.79 | <u>18.06 (14.46)</u> | 18.17 (13.65) | 19.11 (14.36) | 30.10 (21.19) | 21.36 (15.92) |
| 12  | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[+ E2E-FT with simu (152h)](exp/Exp10~12-CUSIDE-Array+Pre-trained_BE+E2E-FT(ID+simu_data)) | 85.79 | <u>20.67 (14.62)</u> | 21.54 (13.93) | 22.26 (14.61) | 33.27 (21.39) | 24.44 (16.14) |
| 13  | [Alimeeting-FE + Pre-trained BE plug in](exp/Exp13-CUSIDE-Array(OOD)+Pre-trained_BE/readme.md) | 85.79 | 35.97 (21.76) | <u>33.32 (17.90)</u> | <u>34.98 (19.31)</u> | 35.84 (24.75) | 35.03 (20.93) |
| 14  | [MFCCA (w/o LM)](https://www.modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary) | 47.06 | <u>$\square$ (21.69)</u> | <u>$\square$ (12.80)</u> | <u>$\square$ (13.97)</u> | $\square$ (61.79) | $\square$ (27.56) |


