# LLM-based phoneme-to-grapheme for phoneme-based speech recognition

This is the official code for the INTERSPEECH 2025 paper "[LLM-based phoneme-to-grapheme for phoneme-based speech recognition](https://arxiv.org/abs/2506.04711)". We release the code and the models. You can find them in their corresponding experimental folders. The pipeline of data processing follows [lang-process.md](../cv-lang10/lang-process/lang-process.md) in Whistle. Below is a brief instruction for each folder and a summary of the experimental results. 

Note that beyond the INTERSPEECH 2025 paper, here we show several extended experiments, including SKM, resource comparison, and significance testing.

## data
[`data`](./data/) contains the efficient data management file [metainfo.json](./data/metainfo.json).


## local
[`local`](./local/) contains some scripts for LLM-P2G method. 

## exp
[`exp`](./exp/) contains configuration files and detailed training process of our models.
### Experiments
Experiments are organized as follows:
- The P2G baselines are the two crosslingual fine-tuning (FT) models -- phoneme FT and subword FT. 
- Two training strategies of LLM-P2G were applied for comparison, namely DANP and TKM. 


#### [Crosslingual](./exp/crosslingual/crosslingual.md)
We fine-tune the [Whistle_phoneme_S](../cv-lang10/exp/Multilingual/Multi._phoneme_S/readme.md) model over two Commonvoice languages (Polish and German) as baselines of our LLM-P2G. The crosslingual experimental setup follows [Whistle_Crosslingual](../cv-lang10/exp/crosslingual/crosslingual.md).
#### [DANP (Data Augmentation via Noisy Phonemes)](./exp/danp/danp.md)
For DANP, the fine-tuned [Whistle-S2P](./exp/crosslingual/crosslingual.md) is only used to generate noisy phonemes by beam search decoding or sampling before LLM-P2G training. K is the beam width or sample size, which is also the hyper-parameter for the amount of data augmentation. After generating noisy phonemes, LLM-P2G is fine-tuned with 9 different DANP strategies for ablation study, including K is set to be from 1 to 96 and multiple model checkpoints data.
#### [TKM (Top-K Marginalized)](./exp/tkm/tkm.md)
For TKM, the fine-tuned [Whistle-S2P](./exp/crosslingual/crosslingual.md), being frozen, is used to generate phoneme candidates in real time by beam search in training and decoding. K is the beam width, which is also the hyper-parameter for the number of phoneme sequence candidates. LLM-P2G is fine-tuned with 4 different TKM strategies for ablation study, including K is set to be from 8 and 32 and random 8 of top-32.
#### [SKM (Sampling-K Marginalized)](./exp/skm/skm.md)
For SKM, we replace beam searching decoding with sampling and further introduce a temperature factor (T) to control the randomness of generation. This reduces computational cost while maintaining phoneme diversity for marginalized training and decoding. T is set to be 1.5.


### Results
#### Comparison of WERs for LLM-P2G and baselines (WFST-based models) on 130 hours Polish and German data
| Model | Polish | | German | |
| ------ | ------ | ------ | ------ | ------ | 
|  | w/o LM | w LM | w/o LM | w LM |
| [Whistle phoneme FT](./exp/crosslingual/crosslingual.md) | - | 4.30 | - | 15.73 | 
| [Whistle subword FT](./exp/crosslingual/crosslingual.md) | 5.84 | 3.82 | 14.09 | 14.01 | 
| [LLM-P2G](./exp/danp/danp.md) | 5.71 | 5.04 | 14.76 | 14.39 | 
| [LLM-P2G + DANP](./exp/danp/danp.md) | 4.44 | 4.18 | 13.86 | 13.63 | 
| [LLM-P2G + randomized TKM](./exp/tkm/tkm.md) | 4.01 | 3.68 | 13.44 | 13.03 | 
| [LLM-P2G + SKM](./exp/skm/skm.md) | __3.98__ | __3.61__ | __13.21__ | __12.94__ |

#### Comparison of WERs for LLM-P2G and baselines (WFST-based models) on 20 hours Polish and German data
| Model | Polish | | German | |
| ------ | ------ | ------ | ------ | ------ | 
|  | w/o LM | w LM | w/o LM | w LM |
| [Whistle phoneme FT](./exp/crosslingual/crosslingual.md) | - | 16.27 | - | 30.71 | 
| [Whistle subword FT](./exp/crosslingual/crosslingual.md) | __17.59__ | __13.84__ | __27.78__ | __28.04__ | 
| [LLM-P2G](./exp/danp/danp.md) | 23.75 | 21.56 | 32.26 | 31.45 | 
| [LLM-P2G + DANP](./exp/danp/danp.md) | 19.99 | 19.05 | 30.49 | 29.97 | 
| [LLM-P2G + randomized TKM](./exp/tkm/tkm.md) | 19.19 | 17.36 | 29.20 | 28.78 |

### Ablation study
#### Consumption of CPU memory, GPU memory, disk storage and real time factor in decoding process for LLM-P2G and WFST-based models
We compare the decoding resource consumption of LLM-P2G and WFST-based methods to evaluate their efficiency and deployment profile for real-world applications. Specifically, we measure CPU memory usage with the @profile decorator, monitor GPU memory through real-time outputs during debugging, and use the time command in bash to record decoding duration. Besides, we calculate the real-time factor (RTF) and compare the CPU, GPU, and storage overhead of both methods.
| Model | CPU consumption | GPU consumption | Storage consumption| RTF |
| ------ | ------ | ------ | ------ | ------ |
| Whistle phoneme FT | __4.0 GB__ | __1.57 GB__ | __0.7 GB__ | __0.02__ | 
| Whistle subword FT | 5.9 GB | __1.57 GB__ | 2.4 GB | 0.07 | 
| LLM-P2G + DANP | __4.0 GB__ | 4.6 GB | 2.5 GB | 0.07 | 
| LLM-P2G + randomized TKM | __4.0 GB__ | 6.3 GB | 2.5 GB | 0.10 | 

#### Significance test for the best results of LLM-P2G and WFST-based models
To evaluate whether the performance difference between the LLM-P2G and WFST-based methods is statistically significant, we conduct a matched-pair significance test using Word Error Rate (WER) as the evaluation metric. Run the script [significance_test.py](https://github.com/thu-spmi/CAT/blob/trf/cat/utils/significance_test.py) to compute the p-value.
| Language | p-value | 
| ------ | ------ | 
| Polish (3.68 vs. 3.82) | 1e-04 | 
| German (13.03 vs. 14.01) | 7e-23 | 

