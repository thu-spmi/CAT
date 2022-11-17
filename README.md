<div align="center"><img src="./assets/logo.png" width=200></div>

# CAT: Crf-based Asr Toolkit
**CAT provides a complete workflow for CRF-based data-efficient end-to-end speech recognition.**

- [Overview](#overview)
- [Features](#features)
- [Getting started](#getting-started)
- [Further reading](#further-reading)
- [Publications](#publications)

## Overview

CAT aims at combining the advantages of both the hybrid and the E2E ASR systems (in terms of modularity versus unified neural network, separate optimization versus joint optimization, etc.). CAT advocates discriminative training (and the global normalization model) in the framework of [conditional random field](https://en.wikipedia.org/wiki/Conditional_random_field) (CRF), particularly with but not limited to [connectionist temporal classification](https://mediatum.ub.tum.de/doc/1292048/file.pdf) (CTC) inspired state topology.


## Features

1. CAT contains a full-fledged CUDA/C/C++ implementation of CTC-CRF loss function binding to PyTorch

2. One-stop CTC/CTC-CRF/RNN-T/LM training & inference. See the [templates](egs/TEMPLATE)

3. Flexible configuration with JSON. Check the [guideline for configuration](docs/configure_guide.md)

4. Scalable and extensible. It's easy to be extended to train tens of thousands of speech data and add new models and tasks


## Installation

1. Dependencies

   - CUDA compatible device, NVIDIA driver installed and CUDA lib available.
   - PyTorch: `>=1.9.0` is required. [Installation guide from PyTorch](https://pytorch.org/get-started/locally/#start-locally)
   - [Kaldi](https://github.com/kaldi-asr/kaldi) **\[optional, but recommended\]**: used for speech data preparation and some FST-related operations. This is optional for most of the basic functions. Only if you want to conduct [CTC-CRF](egs/TEMPLATE/exp/asr-ctc-crf) training, this is required.
      
      Besides Kaldi, you could use `torchaudio` for feature extraction. Take a look at [data.sh](egs/aishell/local/data.sh) for how to for preparing data with `torchaudio`.

2. Clone and install CAT

   ```bash
   git clone https://github.com/thu-spmi/CAT.git && cd CAT
   # Get installation helping message
   ./install.sh -h
   # Install with default configurations
   #./install.sh
   ```

## Getting started

To get started with this project, please refer to [TEMPLATE](egs/TEMPLATE/README.md) for tutorial.

## Further reading

- [Guideline for configuring settings](docs/configure_guide.md)
- [Tutorial for CUSIDE](docs/cuside_ch.md) \[Chinese|中文\]: learn to run experiment with [CUSIDE](https://arxiv.org/abs/2203.16758)
- [Some tips about the usage of third party tools](docs/guide_for_third_party_tools.md)
- [Guide to train models on more than 1500 hours of speech data](docs/how_to_prepare_large_dataset_ch.md) \[Chinese|中文\]
- [Changelog](docs/changelog.md)

## ASR results

Some of the results are obtained on CAT v2.

| dataset                                                                                                                    | evaluation sets         | performance  |
| -------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ------------ |
| [AISHELL-1](https://github.com/thu-spmi/CAT/tree/v3-dev/egs/aishell#result)                                                | dev / test              | 4.25 / 4.47  |
| [Commonvoice German](https://github.com/thu-spmi/CAT/blob/master/egs/commonvoice/RESULT.md#conformertransformer-rescoring) | test                    | 9.8          |
| [Librispeech](https://github.com/thu-spmi/CAT/tree/v3-dev/egs/libri#result)                                                | test-clean / test-other | 1.94 / 4.39  |
| [Switchboard](https://github.com/thu-spmi/CAT/blob/master/egs/swbd/RESULT.md#conformertransformer-rescoring)               | switchboard / callhome  | 6.9 / 14.5   |
| [THCHS30](https://github.com/thu-spmi/CAT/blob/master/egs/thchs30/RESULT.md#vgg-blstm)                                     | test                    | 6.01         |
| [Wenetspeech](https://github.com/thu-spmi/CAT/tree/v3-dev/egs/wenetspeech#result)                                          | test-net / test-meeting | 9.32 / 14.66 |
| [WSJ](https://github.com/thu-spmi/CAT/blob/master/egs/wsj/RESULT.md)                                                       | eval92 / dev93          | 2.77 / 5.68  |


## Publications

