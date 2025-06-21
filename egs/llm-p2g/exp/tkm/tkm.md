# Top-K marginalized (TKM) method for LLM-P2G
Beam search decoding is also used to generate noisy phonemes for marginalized training and decoding. A key feature of TKM is that the generation of these noisy phoneme sequences is performed on-the-fly during training, rather than through offline preprocessing. We also fine-tune the LLM-P2G (mT5-base) with different TKM strategies. Notably, due to training requirements, I made some modifications to the forward function of the official mT5 implementation. Details can be found in the [forward_MT5ForConditionalGeneration.py](../../local/forward_MT5ForConditionalGeneration.py).


## TKM strategies
| Strategy | Phoneme data for marginalization | Checkpoint for S2P decoding | Hyper-p in `config.json` |
| ------ | ------ | ------ | ------ |
| top-8-beam |  top-8 phoneme beam results| best-3.pt of Whistle-S2P-130h | sample_beam=8, beam_width=8 |
| top-32-beam | top-32 phoneme beam results| best-3.pt of Whistle-S2P-130h | sample_beam=32, beam_width=32 |
| random-8-of-32-beam | random 8 out of 32 phoneme beam results| best-3.pt of Whistle-S2P-130h | sample_beam=8, beam_width=32 |
| random-8-of-32-beam_20h | random 8 out of 32 phoneme beam results| best-3.pt of Whistle-S2P-20h | sample_beam=8, beam_width=32 |



## Results
### WER of LLM-P2G with TKM

| TKM strategy | Polish | | German | |
| ------ | ------ | ------ | ------ | ------ |
| | w/o LM | w LM | w/o LM | w LM |
| random-8-of-32-beam_20h | [19.19](../tkm/pl/random-8-of-32-beam_20h/readme.md) | [17.36](../tkm/pl/random-8-of-32-beam_20h/readme.md) | [29.20](../tkm/de/random-8-of-32-beam_20h/readme.md) | [28.78](../tkm/de/random-8-of-32-beam_20h/readme.md) |
| top-32-beam | [16.55](../tkm/pl/top-32-beam/readme.md) | [16.12](../tkm/pl/top-32-beam/readme.md) | [21.69](../tkm/de/top-32-beam/readme.md) | [21.31](../tkm/de/top-32-beam/readme.md) |
| top-8-beam | [4.31](../tkm/pl/top-8-beam/readme.md) | [3.80](../tkm/pl/top-8-beam/readme.md) | [13.58](../tkm/de/top-8-beam/readme.md) | [13.18](../tkm/de/top-8-beam/readme.md) |
| random-8-of-32-beam | [4.01](../tkm/pl/random-8-of-32-beam/readme.md) | [3.68](../tkm/pl/random-8-of-32-beam/readme.md) | [13.44](../tkm/de/random-8-of-32-beam/readme.md) | [13.03](../tkm/de/random-8-of-32-beam/readme.md) |


