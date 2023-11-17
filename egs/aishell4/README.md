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


**Main results**

Evaluated by CER (%)

| EXP ID              | dev  | test | notes                       |
| ------------------- |:----:|:----:| --------------------------- |
| [bf+ctc](exp/ctc-e2e-chunk)| 20.22 | 29.65 | non-streaming |
| [bf+ctc](exp/ctc-e2e-chunk)| 20.96 | 30.73 | streaming |
| [bf+ctc+simu](exp/ctc-e2e-chunk+simu)| 21.13 | 29.78 | non-streaming|
| [bf+ctc+simu](exp/ctc-e2e-chunk+simu)| 23.62 | 32.98 | streaming |



