# Sampling K marginalized (SKM) method for LLM-P2G
While [TKM](../tkm/tkm.md) relies on beam search decoding to produce noisy phonemes, SKM replaces this with sampling and further introduces a temperature factor (T) to control the randomness of generation. This reduces computational cost while maintaining phoneme diversity for marginalized training and decoding. Apart from these two differences, SKM shares the same configuration as TKM. Notably, due to training requirements, I made some modifications to the forward function of the official mT5 implementation. Details can be found in the [forward_MT5ForConditionalGeneration.py](../../local/forward_MT5ForConditionalGeneration.py).


## SKM strategies
| Strategy | Phoneme data for marginalization | Checkpoint for S2P sampling | Hyper-p in `config.json` |
| ------ | ------ | ------ | ------ |
| 8-sample-T | 8 Sampling phoneme results from logits with high temperature | best-3.pt of Whistle-S2P-130h | sample_size=8, sample_beam=8, T_weight_s2p=1.5 |



## Results
### WER of LLM-P2G with SKM

| SKM strategy | Polish | | German | |
| ------ | ------ | ------ | ------ | ------ |
| | w/o LM | w LM | w/o LM | w LM |
| random-8-sample-T | [3.98](../SKM/pl/random-8-sample-T/readme.md) | [3.61](../SKM/pl/random-8-sample-T/readme.md) | [13.21](../SKM/de/random-8-sample-T/readme.md) | [12.94](../SKM/de/random-8-sample-T/readme.md) |


