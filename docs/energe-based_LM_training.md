# Energy-based language model training

This is the official code for the paper "Exploring Energy-based Language Models with Different Architectures and Training Methods for Speech Recognition".

## Installation

Please refer to the instruction of [CAT](https://github.com/thu-spmi/CAT/) for installation.

## Experiments

We explore energy-based language models (ELMs) with different architectures and training methods, all using large pretrained transformers as backbones. Performance of the ELMs is evaluated via 2-pass rescoring on ASR recognized N-best list. Experiments are conducted on two widely-used ASR datasets: AISHELL-1, WenetSpeech. Please refer to corresponding directories for specific instructions.

| Datasets    | Architecture | Training Method | Link                                                    |
| ----------- | ------------ | --------------- | ------------------------------------------------------- |
| AISHELL-1   | GN-ELM       | DNCE            | [exp1](../egs/aishell/exp/ebm-lm/GN-ELM-DNCE/readme.md) |
| AISHELL-1   | GN-ELM       | MLE             | [exp2](../egs/aishell/exp/ebm-lm/GN-ELM-ML/readme.md)   |
| AISHELL-1   | GN-ELM       | NCE             | [exp3](../egs/aishell/exp/ebm-lm/GN-ELM-NCE/readme.md)  |
| AISHELL-1   | TRF-LM       | DNCE            | [exp4](../egs/aishell/exp/ebm-lm/TRF-LM-DNCE/readme.md) |
| WenetSpeech | GN-ELM       | DNCE            | [exp5](../egs/wenetspeech/exp/lm/GN-ELM-DNCE/readme.md) |
| WenetSpeech | TRF-LM       | DNCE            | [exp6](../egs/wenetspeech/exp/lm/TRF-LM-DNCE/readme.md) |

### Notes

* The specific meanings of the terms in the table are explained in our paper.
* The neural architecture of the ELM can be choosen from 3 different types.

  - `SumTargetLogit`: set `config_ebm['decoder']['kwargs']['model_name']=GPT2LMHeadModel` and `config['decoder']['kwargs']['energy_func']='sumtargetlogit'`;
  -  `Hidden2Scalar`: set `config_ebm['decoder']['kwargs']['model_name']=BertModel` and `config['decoder']['kwargs']['energy_func']='hidden2scalar-sum'`;
  -  `SumTokenLogit`: set `config_ebm['decoder']['kwargs']['model_name']=BertLMHeadModel` and `config['decoder']['kwargs']['energy_func']='sumtokenlogit'`;

### Significance Test

If you want to determine whether there is a significant difference between two models statistically, please refer to [significance test](./significance_test.md).