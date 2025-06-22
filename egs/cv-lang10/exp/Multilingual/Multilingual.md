# Multilingual ASR model
### Training data
All of our multilingual ASR model are trained with 10 languages of cv-lang10, which has been processed in [lang-process](../../lang-process/lang-process.md). But for English wav2vec-base model and multilingul wav2vec-base model, only audio are used to train the model. The language ID and training hours of the ten languages are in the following table.

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

### Model
Five multilingual ASR model are trained with our cv-lang10 data. 
* The [Multi. phoneme S](./Multi._phoneme_S/readme.md) model and [Multi. subword S](./Multi._subword/readme.md) model are based on the Conformer network consisting of 14 encoder blocks with 4 heads and 36 dimensions hidden states, followed by a 512 dimensions feed-forward network. The Noam optimizer was adopted and warm-up steps were set to 10% of training steps.
* The [Multi. phoneme M](./Multi._phoneme_M/readme.md) model is based on the Conformer network consisting of 22 encoder blocks with 4 heads and 160 dimensions hidden states, followed by a 640 dimensions feed-forward network. The Noam optimizer was adopted and warm-up steps were set to 10% of training steps.
* The [Multi. phoneme L](./Multi._phoneme_L/readme.md) model is based on the Conformer network consisting of 22 encoder blocks with 4 heads and 224 dimensions hidden states, followed by a 1024 dimensions feed-forward network. The Noam optimizer was adopted and warm-up steps were set to 10% of training steps.
* The [Multi. wav2vec-base](./Multi._wav2vec-base/readme.md) model is trained with the sequence-to-sequence toolkit [faieseq](https://github.com/facebookresearch/fairseq). And the model architecture follows [wav2vec-base](https://huggingface.co/facebook/wav2vec2-base/tree/main).

### Results
* %PER
    | Model | Model size | en | es | fr | it | ky | nl | ru | sv-SE | tr | tt | Avg.
    | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
    | [Multi. phoneme S](./Multi._phoneme_S/readme.md) | 90 MB | 8.02 | 3.37 | 5.68 | 4.04 | 8.29 | 5.77 | 6.05 | 18.07 | 8.32 | 8.53 | 7.61 |
    | [Multi. phoneme M](./Multi._phoneme_M/readme.md) | 218 MB | 6.70 | 2.63 | 4.53 | 3.12 | 5.95 | 3.95 | 4.61 | 14.81 | 6.04 | 8.47 | 6.08 |
    | [Multi. phoneme L](./Multi._phoneme_L/readme.md) | 543 MB | __5.42__ | __1.96__ | __3.52__ | __2.25__ | __4.06__ | __2.64__ | __2.97__ | __11.33__ | __4.04__ | __5.97__ | __4.41__ |

* %WER with 4-gram LM 
    | Model | Model size | en | es | fr | it | ky | nl | ru | sv-SE | tr | tt | Avg.
    | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
    | [Multi. subword](./Multi._subword/readme.md) | 92 MB | 12.00 | 9.82 | 12.40 | 9.98 | 3.29 | 9.67 | 3.31 | 9.95 | 9.11 | 13.56 | 9.30 |
    | [Multi. phoneme S](./Multi._phoneme_S/readme.md) | 90 MB | 10.76 | 8.68 | 16.01 | 9.98 | 1.02 | 7.32 | 1.59 | 6.14 | 7.63 | 7.30 | 7.64 |
    | [Multi. phoneme M](./Multi._phoneme_M/readme.md) | 218 MB | 9.83 | 7.82 | 14.94 | 9.04 | __0.91__ | 6.57 | 1.65 | 5.65 | 7.27 | 7.37 | 7.10 |
    | [Multi. phoneme L](./Multi._phoneme_L/readme.md) | 543 MB | __8.80__ | __7.02__ | __14.02__ | __8.16__ | 0.94 | __6.22__ | __1.46__ | __5.06__ | __7.05__ | __6.92__ | __6.56__ |