# Whistle: Data-Efficient Multilingual and Crosslingual Speech Recognition via Weakly Phonetic Supervision

This is the official code for the paper "[Whistle: Data-Efficient Multilingual and Crosslingual Speech Recognition via Weakly Phonetic Supervision](https://arxiv.org/abs/2406.02166)". We release the code,
models and data for the whole pipeline of Whistle and you can find them in their respective experimental folders. Below is a brief instruction for each folder and a summary of the experimental results.

## data
[`data`](./data/) contains the efficient data management file [metainfo.json](./data/metainfo.json).

## lang-process
All of our ASR models are trained with the processed CV-lang10 data covering 12 languages(10 seen languages and 2 unseen languages), which are sourced from the publicly available [`Common Voice`](https://commonvoice.mozilla.org/) 11.0. The data processing for each language are detailed in [`lang-process.md`](./lang-process/lang-process.md). For convenience, we adopt ISO-639-1 code to represent language ID, and the 12 languages and training hours are as follow.
| Serial number | Language | Language ID | Training hours |
| ----------- | ----------- | ----------- | ----------- |
| 1 | `English` | `en` | 2227.3 |
| 2 | `Spanish` | `es` | 382.3 |
| 3 | `French` | `fr` | 823.4 |
| 4 | `Italian` | `it` | 271.5 |
| 5 | `Kyrgyz` | `ky` | 32.7 |
| 6 | `Dutch` | `nl` | 70.2 |
| 7 | `Russian` | `ru` | 149.8 |
| 8 | `Swedish` | `sv-SE` | 29.8 |
| 9 | `Turkish` | `tr` | 61.5 |
| 10 | `Tatar` | `tt` | 20.8 |
| 11 | `Polish` | `pl` | 130 |
| 12 | `Indonesian` | `id` | 20.8 |

## local
[`local`](./local/) contains the script [data_prep.md](./local/data_prep.md) preparing data and generating pronunciation lexicon for each language. Besides, there are some useful tools to debug our experiments.

## exp
[`exp`](./exp/) contains configuration files and detailed training process of our models.
### Experiment setup
We adapt the Conformer and CTC to train our models. Three training strategies were applied for comparison, which are monolingual, multilingual and cross-lingual training.

#### [Monolingual](./exp/Monolingual/readme.md)
10 monolingual phoneme-based ASR models are trained on each language dataset seperately and then is evaluated on test dataset of corresponding language whitout fine-tuneing. For Indonesian and Polish, the training data is divided into three scales: 1 hour, 10 hours, and full. And the phoneme-based model and subword-based model are both trained with these scales data seperately.
#### [Multilingual](./exp/Multilingual/readme.md)
3 phoneme-based models of different sizes are trained, including small(90 MB), medium(218 MB) and large(543 MB). And subword-based and wav2vec-based model of small size are also trained for comprison. The multilingual ASR model are trained on CV-lang10 data and then is evaluated on test dataset of corresponding language whitout fine-tuneing.
#### [Crosslingual](./exp/Crosslingual/readme.md)
To test different multilingual pre-trained models for crosslingual speech recognition, we conduct phoneme-based and subword-based crosslingual fine-tuning on unseen languages. All of the Crosslingual models are fine-tuned on the basis of the pretrained multilingual phoneme-based model of small size, subword-based model or wav2vec-based model with the same fine-tuning strategy. The performence of the fine-tuned models are evaluated on 2 unseen languages dataset.

### Results
#### Phoneme based monolingial models and multilingual pretrained models (PER%) 
| Model | Model size | en | es | fr | it | ky | nl | ru | sv-SE | tr | tt | Avg.
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [Monolingual phoneme](./exp/Monolingual/readme.md) | 90 MB | 7.92 | 2.47 | 4.93 | 2.87 | 2.23 | 5.89 | 2.72 | 16.11 | 6.00 | 10.54 | 6.16 |
| [Multilingual phoneme small](./exp/Multilingual/Multi._phoneme_S/readme.md) | 90 MB | 8.02 | 3.37 | 5.68 | 4.04 | 8.29 | 5.77 | 6.05 | 18.07 | 8.32 | 8.53 | 7.61 |
| [Multilingual phoneme medium](./exp/Multilingual/Multi._phoneme_M/readme.md) | 218 MB | 6.70 | 2.63 | 4.53 | 3.12 | 5.95 | 3.95 | 4.61 | 14.81 | 6.04 | 8.47 | 6.08 |
| [Multilingual phoneme large](./exp/Multilingual/Multi._phoneme_S/readme.md) | 543 MB | 5.42 | 1.96 | 3.52 | 2.25 | 4.06 | 2.64 | 2.97 | 11.33 | 4.04 | 5.97 | 4.43 |

#### Phoneme based monolingial models and multilingual pretrained models(WER%)
| Model | Model size | en | es | fr | it | ky | nl | ru | sv-SE | tr | tt | Avg.
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [Monolingual phoneme](./exp/Monolingual/readme.md) | 90 MB | 10.59 | 7.91 | 15.58 | 9.26 | 1.03 | 8.84 | 1.62 | 8.37 | 8.46 | 9.75 | 8.14 |
| [Multilingual subword small](./exp/Multilingual/Multi._subword/readme.md) | 92 MB | 12.00 | 9.82 | 12.40 | 9.98 | 3.29 | 9.67 | 3.31 | 9.95 | 9.11 | 13.56 | 9.30 |
| [Multilingual phoneme small](./exp/Multilingual/Multi._phoneme_S/readme.md) | 90 MB | 10.76 | 8.68 | 16.01 | 9.98 | 1.02 | 7.32 | 1.59 | 6.714 | 7.63 | 7.30 | 7.64 |
| [Multilingual phoneme medium](./exp/Multilingual/Multi._phoneme_M/readme.md) | 218 MB | 9.83 | 7.82 | 14.94 | 9.04 | 0.91 | 6.57 | 1.65 | 5.65 | 7.27 | 7.37 | 7.10 |
| [Multilingual phoneme large](./exp/Multilingual/Multi._phoneme_S/readme.md) | 543 MB | 8.80 | 7.02 | 14.02 | 8.16 | 0.94 | 6.22 | 1.46 | 5.06 | 7.05 | 6.92 | 6.56 |


#### Phoneme based crosslingual fine-tuning on Polish (WER%)
| Model | 10 minutes | 1 hour | 10 hours | 130 hours (full)|
| ------ | ------ | ------ | ------ | ------ |
| Monolingual phoneme | - | [99.98](./exp/Monolingual/pl/Mono._phoneme_1h/readme.md) | [13.86](./exp/Monolingual/pl/Mono._phoneme_10h/readme.md) | [4.97](./exp/Monolingual/pl/Mono._phoneme_130h/readme.md) |
| Wav2vec (En) phoneme FT | - | [11.09](./exp/Crosslingual/pl/Wav2vec-En_ft_phoneme_1h/readme.md) | [6.75](./exp/Crosslingual/pl/Wav2vec-En_ft_phoneme_10h/readme.md) | [4.57](./exp/Crosslingual/pl/Wav2vec-En_ft_phoneme_130h/readme.md) |
| Wav2vec (10 lang) phoneme FT | - | [7.94](./exp/Crosslingual/pl/Wav2vec-lang10_ft_phoneme_1h/readme.md) | [5.65](./exp/Crosslingual/pl/Wav2vec-lang10_ft_phoneme_10h/readme.md) | [4.44](./exp/Crosslingual/pl/Wav2vec-lang10_ft_phoneme_130h/readme.md) |
| Phoneme PT and phoneme FT | [11.0](./exp/Crosslingual/pl/Multi._phoneme_ft_phoneme_10m/readme.md) | [6.95](./exp/Crosslingual/pl/Multi._phoneme_ft_phoneme_1h/readme.md) | [5.27](./exp/Crosslingual/pl/Multi._phoneme_ft_phoneme_10h/readme.md) | [4.30](./exp/Crosslingual/pl/Multi._phoneme_ft_phoneme_130h/readme.md) |


#### Subword based crosslingual fine-tuning on Polish (WER%)
| Model | 10 minutes | 1 hour | 10 hours | 130 hours (full)|
| ------ | ------ | ------ | ------ | ------ |
| Monolingual subword | - | [98.38](./exp/Monolingual/pl/Mono._subword_1h/readme.md) | [59.43](./exp/Monolingual/pl/Mono._subword_10h/readme.md) | [7.12](./exp/Monolingual/pl/Mono._subword_130h/readme.md) |
| Wav2vec (En) subword FT | - | [100](./exp/Crosslingual/pl/Wav2vec-En_ft_subword_1h//readme.md) | [7.08](./exp/Crosslingual/pl/Wav2vec-En_ft_subword_10h/readme.md) | [3.85](./exp/Crosslingual/pl/Wav2vec-En_ft_subword_130h/readme.md) |
| Wav2vec (10 lang) subword FT | - | [100](./exp/Crosslingual/pl/Wav2vec-lang10_ft_subword_1h/readme.md) | [5.71](./exp/Crosslingual/pl/Wav2vec-lang10_ft_subword_10h/readme.md) | [3.45](./exp/Crosslingual/pl/Wav2vec-lang10_ft_subword_130h/readme.md) |
| Subword PT and subword FT | [52.52](./exp/Crosslingual/pl/Multi._subword_ft_subword_10m/readme.md) | [9.16](./exp/Crosslingual/pl/Multi._subword_ft_subword_1h/readme.md) | [4.89](./exp/Crosslingual/pl/Multi._subword_ft_subword_10h/readme.md) | [3.76](./exp/Crosslingual/pl/Multi._subword_ft_subword_130h/readme.md) |
| Phoneme PT and subword FT | [81.62](./exp/Crosslingual/pl/Multi._phoneme_ft_subword_10m/readme.md) | [8.63](./exp/Crosslingual/pl/Multi._phoneme_ft_subword_1h/readme.md) | [4.83](./exp/Crosslingual/pl/Multi._phoneme_ft_subword_10h/readme.md) | [3.82](./exp/Crosslingual/pl/Multi._phoneme_ft_subword_130h/readme.md) |


#### Phoneme based crosslingual fine-tuning on Indonesian (WER%)
| Model | 10 minutes | 1 hour | 10 hours | 20 hours (full) |
| ------ | ------ | ------ | ------ | ------ |
| Monolingual phoneme | - | [100](./exp/Monolingual/id/Mono._phoneme_1h/readme.md) | [7.71](./exp/Monolingual/id/Mono._phoneme_10h/readme.md) | [3.28](./exp/Monolingual/id/Mono._phoneme_20h/readme.md) |
| Wav2vec (En) phoneme FT | - | [6.73](./exp/Crosslingual/id/Wav2vec-En_ft_phoneme_1h/readme.md) | [3.31](./exp/Crosslingual/id/Wav2vec-En_ft_phoneme_10h/readme.md) | [2.83](./exp/Crosslingual/id/Wav2vec-En_ft_phoneme_2h/readme.md) |
| Wav2vec (10 lang) phoneme FT | - | [3.75](./exp/Crosslingual/id/Wav2vec-lang10_ft_phoneme_1h/readme.md) | [2.79](./exp/Crosslingual/id/Wav2vec-lang10_ft_phoneme_10h/readme.md) | [2.47](./exp/Crosslingual/id/Wav2vec-lang10_ft_phoneme_20h/readme.md) |
| Phoneme PT and phoneme FT | [6.85](./exp/Crosslingual/id/Multi._phoneme_ft_phoneme_10m/readme.md) | [3.27](./exp/Crosslingual/id/Multi._phoneme_ft_phoneme_1h/readme.md) | [2.54](./exp/Crosslingual/id/Multi._phoneme_ft_phoneme_10h/readme.md) | [2.43](./exp/Crosslingual/id/Multi._phoneme_ft_phoneme_20h/readme.md) |

#### Subword based crosslingual fine-tuning on Indonesian (WER%)
| Model | 10 minutes | 1 hour | 10 hours | 20 hours (full)|
| ------ | ------ | ------ | ------ | ------ |
| Monolingual subword | - | [96.42](./exp/Monolingual/id/Mono._subword_1h/readme.md) | [49.67](./exp/Monolingual/id/Mono._subword_10h/readme.md) | [10.85](./exp/Monolingual/id/Mono._subword_20h/readme.md) |
| Wav2vec (En) Subword FT | - | [100](./exp/Crosslingual/id/Wav2vec-En_ft_subword_1h/readme.md) | [5.28](./exp/Crosslingual/id/Wav2vec-En_ft_subword_10h/readme.md) | [3.59](./exp/Crosslingual/id/Wav2vec-En_ft_subword_20h/readme.md) |
| Wav2vec (10 lang) Subword FT | - | [99.97](./exp/Crosslingual/id/Wav2vec-lang10_ft_subword_1h/readme.md) | [4.52](./exp/Crosslingual/id/Wav2vec-lang10_ft_subword_10h/readme.md) | [3.15](./exp/Crosslingual/id/Wav2vec-lang10_ft_subword_20h/readme.md) |
| Subword PT and subword FT | [87.75](./exp/Crosslingual/id/Multi._subword_ft_subword_10m/readme.md) | [23.56](./exp/Crosslingual/id/Multi._subword_ft_subword_1h/readme.md) | [3.91](./exp/Crosslingual/id/Multi._subword_ft_subword_10h/readme.md) | [3.07](./exp/Crosslingual/id/Multi._subword_ft_subword_20h/readme.md) |
| Phoneme PT and subword FT | [98.65](./exp/Crosslingual/id/Multi._phoneme_ft_subword_10m/readme.md) | [24.57](./exp/Crosslingual/id/Multi._phoneme_ft_subword_1h/readme.md) | [3.59](./exp/Crosslingual/id/Multi._phoneme_ft_subword_10h/readme.md) | [2.92](./exp/Crosslingual/id/Multi._phoneme_ft_subword_20h/readme.md) |
