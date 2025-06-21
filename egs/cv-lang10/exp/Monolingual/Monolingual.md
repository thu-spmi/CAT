# Monolingual ASR model
## Monolingual ASR model with full data
We trained a phoneme-based ASR model for each language of cv-lang10 with the same architecture that is based on a Conformer network consisting of 14 encoder blocks. The number of phonemes and training hours of the each language are in the following table.

| Language | Language ID | # of phonemes | Train hours | Dev hours | Test hours |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| `English` | `en` | 39 | 2227.3 | 27.2 | 27.0 |
| `Spanish` | `es` | 32 | 382.3 | 26.0 | 26.5 |
| `French` | `fr` | 33 | 823.4 | 25.0 | 25.4 |
| `Italian` | `it` | 30 | 271.5 | 24.7 | 26.0 |
| `Kirghiz` | `ky` | 32 | 32.7 | 2.1 | 2.2 |
| `Dutch` | `nl` | 39 | 70.2 | 13.8 | 13.9 |
| `Russian` | `ru` | 32 | 149.8 | 14.6 | 15.0 |
| `Swedish` | `sv-SE` | 33 | 29.8 | 5.5 | 6.2 |
| `Turkish` | `tr` | 41 | 61.5 | 10.1 | 11.4 |
| `Tatar` | `tt` | 31 | 20.8 | 3.0 | 5.7 |

* %PER
    | Model | Model size | en | es | fr | it | ky | nl | ru | sv-SE | tr | tt | Avg.
    | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
    | Mono. phoneme | 90 MB | [7.39](./en/readme.md) | [2.47](./es/readme.md) | [4.93](./fr/readme.md) | [2.87](./it/readme.md) | [2.23](./ky/readme.md) | [4.60](./nl/readme.md) | [2.72](./ru/readme.md) | [18.69](./sv-SE/readme.md) | [6.00](./tr/readme.md) | [10.54](./tt/readme.md) | 6.11 |

* %WER with 4-gram LM 
    | Model | Model size | en | es | fr | it | ky | nl | ru | sv-SE | tr | tt | Avg.
    | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
    | Mono. phoneme | 90 MB | [10.59](./en/readme.md) | [7.91](./es/readme.md) | [15.58](./fr/readme.md) | [9.26](./it/readme.md) | [1.03](./ky/readme.md) | [8.84](./nl/readme.md) | [1.62](./ru/readme.md) | [8.37](./sv-SE/readme.md) | [8.46](./tr/readme.md) | [9.75](./tt/readme.md) | 8.14 |


## Monolingual ASR model with low-resource data
For ablation study, the training data is divided into three scales to simulate different resource scenarios: 1 hour, 10 hours, and full data. Phoneme-based and subword-based models are both trained with this three scales of training data.
| Language | Language ID | # of phonemes | # of subwords | Train hours | Dev hours | Test hours |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| `Indonesian` | `id` | 35 | 500 | 20.8 | 3.7 | 4.1 |
| `Polish` | `pl` | 35 | 500 | 129.9 | 11.4 | 11.5 |

### Phoneme-based
* %PER
    | language | 1 hour | 10 hours | full data |
    | ------ | ------ | ------ | ------ |
    | Indonesian | [96.52](./id/Mono._phoneme_1h/readme.md) | [26.34](./id/Mono._phoneme_10h/readme.md) | [5.74](./id/Mono._phoneme_20h/readme.md) |
    | Polish | [86.01](./pl/Mono._phoneme_1h/readme.md) | [30.38](./pl/Mono._phoneme_10h/readme.md) | [2.82](./pl/Mono._phoneme_130h/readme.md) |

* %WER with LM
    | language | 1 hour | 10 hours | full data |
    | ------ | ------ | ------ | ------ |
    | Indonesian | [100](./id/Mono._phoneme_1h/readme.md) | [9.54](./id/Mono._phoneme_10h/readme.md) | [3.28](./id/Mono._phoneme_20h/readme.md) |
    | Polish | [99.98](./pl/Mono._phoneme_1h/readme.md) | [13.86](./pl/Mono._phoneme_10h/readme.md) | [4.97](./pl/Mono._phoneme_130h/readme.md) |

### Subword-based
* %WER without LM
    | language | 1 hour | 10 hours | full data |
    | ------ | ------ | ------ | ------ |
    | Indonesian | [96.62](./id/Mono._subword_1h/readme.md) | [69.57](./id/Mono._subword_10h/readme.md) | [31.96](./id/Mono._subword_20h/readme.md) |
    | Polish | [98.41](./pl/Mono._subword_1h/readme.md) | [90.98](./pl/Mono._subword_10h/readme.md) | [19.38](./pl/Mono._subword_130h/readme.md) |

* %WER with LM
    | language | 1 hour | 10 hours | full data |
    | ------ | ------ | ------ | ------ |
    | Indonesian | [96.42](./id/Mono._subword_1h/readme.md) | [49.67](./id/Mono._subword_10h/readme.md) | [10.85](./id/Mono._subword_20h/readme.md) |
    | Polish | [98.38](./pl/Mono._subword_1h/readme.md) | [59.43](./pl/Mono._subword_10h/readme.md) | [7.12](./pl/Mono._subword_130h/readme.md) |
