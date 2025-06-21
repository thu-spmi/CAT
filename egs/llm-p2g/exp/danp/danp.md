# Data augmentation with noisy phonemes (DANP) method for LLM-P2G
Beam search decoding and sampling are used to generate noisy phonemes for data augmentation. Specifically, the beam size is from 1 to 96 and the sample size is 500 and 25000 for German and Polish. Using these augmented phoneme data, we fine-tune the LLM-P2G (mT5-base) with different DANP strategies. Notably, due to training requirements, I made some modifications to the forward function of the official mT5 implementation. Details can be found in the [forward_MT5ForConditionalGeneration.py](../../local/forward_MT5ForConditionalGeneration.py).

## DANP strategies
| Strategy | Fine-tuning data | Checkpoint for S2P decoding |
| ------ | ------ | ------ |
| 1-beam | First phoneme beam result| best-3.pt of Whistle-S2P-130h |
| 1-beam_20h | First phoneme beam result| best-3.pt of Whistle-S2P-20h |
| 32-beam | Top-32 phoneme beam results| best-3.pt of Whistle-S2P-130h |
| sampling | Sampling phoneme results| best-3.pt of Whistle-S2P-130h |
| 64-beam | Top-64 phoneme beam results| best-3.pt of Whistle-S2P-130h |
| 32-beam+sampling | Top-32 phoneme beam and sampling results | best-3.pt of Whistle-S2P-130h |
| 96-beam+sampling | Top-96 phoneme beam and sampling results | best-3.pt of Whistle-S2P-130h |
| 96-beam+sampling_multi._ckpts | Top-96 phoneme beam and sampling results | best-3.pt and another 4 checkpoints of Whistle-S2P-130h |
| 96-beam+sampling_multi._ckpts_20h | Top-96 phoneme beam and sampling results | best-3.pt and another 4 checkpoints of Whistle-S2P-20h |

## Generating noisy phonemes
Set the inference mode to `cat.ctc.decode` (default) or `cat.ctc.decode_sample` in the configuration file `hyper-p.json`, and run stage 4 of [run.sh](../../../cv-lang10/run.sh) to decode. Then we run [`read_nbest.py`](../../../../local/read_nbest.py) to obtain the noisy phoneme sequence from decoding results.



## Results
### WER of LLM-P2G with DANP for 130 hours data

| DANP strategy | Polish | | German | |
| ------ | ------ | ------ | ------ | ------ |
| | w/o LM | w LM | w/o LM | w LM |
| 1-beam | [5.71](../danp/pl/1-beam/readme.md) | [5.04](../danp/pl/1-beam/readme.md) | [14.76](../danp/de/1-beam/readme.md) | [14.67](../danp/de/1-beam/readme.md) |
| 32-beam | [4.62](../danp/pl/32-beam/readme.md) | [4.36](../danp/pl/32-beam/readme.md) | [14.17](../danp/de/32-beam/readme.md) | [14.04](../danp/de/32-beam/readme.md) |
| 64-beam | [4.72](../danp/pl/64-beam/readme.md) | [4.36](../danp/pl/64-beam/readme.md) | [14.17](../danp/de/64-beam/readme.md) | [13.97](../danp/de/64-beam/readme.md) |
| 32-beam+sampling | [4.51](../danp/pl/32-beam+sampling/readme.md) | [4.27](../danp/pl/32-beam+sampling/readme.md) | [14.01](../danp/de/32-beam+sampling/readme.md) | [13.91](../danp/de/32-beam+sampling/readme.md) |
| 96-beam+sampling | [4.66](../danp/pl/96-beam+sampling/readme.md) | [4.26](../danp/pl/96-beam+sampling/readme.md) | [13.86](../danp/de/96-beam+sampling/readme.md) | [13.64](../danp/de/96-beam+sampling/readme.md) |
| 96-beam+sampling_multi._ckpts | [4.44](../danp/pl/96-beam+sampling_multi._ckpts/readme.md) | [4.18](../danp/pl/96-beam+sampling_multi._ckpts/readme.md) | [13.86](../danp/de/96-beam+sampling_multi._ckpts/readme.md) | [13.63](../danp/de/96-beam+sampling_multi._ckpts/readme.md) |

### WER of LLM-P2G with DANP for 20 hours data
| DANP strategy | Polish | | German | |
| ------ | ------ | ------ | ------ | ------ |
| | w/o LM | w LM | w/o LM | w LM |
| 1-beam_20h | [23.75](../danp/pl/1-beam_20h/readme.md) | [21.56](../danp/pl/1-beam_20h/readme.md) | [32.26](../danp/de/1-beam_20h/readme.md) | [31.45](../danp/de/1-beam_20h/readme.md) |
| 96-beam+sampling_multi._ckpts_20h | [19.99](../danp/pl/96-beam+sampling_multi._ckpts_20h/readme.md) | [19.05](../danp/pl/96-beam+sampling_multi._ckpts_20h/readme.md) | [30.49](../danp/de/96-beam+sampling_multi._ckpts_20h/readme.md) | [29.97](../danp/de/96-beam+sampling_multi._ckpts_20h/readme.md) |

