# Multilingual wav2vec-base ASR model for 10 languages(Small)
Author: Ma, Te (mate153125@gmail.com)
### Basic info

This model is trained following the architecture of [wav2vec-base](https://huggingface.co/facebook/wav2vec2-base/tree/main) with the sequence-to-sequence toolkit [fairseq](https://github.com/facebookresearch/fairseq). The training dataset(only audio) consists of __4069 hours of `ten languages`__ speech data sourced from the publicly available [`Common Voice`](https://commonvoice.mozilla.org/) 11.0, which are the same as our proposed multilingual model. Run the script [`audio2ark.sh`](../../../../TEMPLATE/local/audio2ark.sh) to prepare training data.

### Training 

* Follow the steps of [Train a Wav2vec 2.0 model with fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md#train-a-wav2vec-20-base-model) to train the model. 

* Fine-tuning on target language is required for wav2vec-based model performance evalution. The more fine-tuning results can be found in [Crosslingual](../../Crosslingual/Crosslingual.md).

### Resource
| Config file | Checkpoint model | Tensorboard log |
| ----------- | ----------- | ----------- |
| [`Wav2vec2_base_cv_lang10.yaml`](./Wav2vec-lang10.yaml) | [`Wav2vec-lang10_best-3.pt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/Wav2vec-lang10/Wav2vec-lang10_best-3.pt) | [`tb_Wav2vec-lang10`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/exp/Wav2vec-lang10/tb_log_Wav2vec-lang10.tar.gz) |

