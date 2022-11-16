# Transducer toolkit for speech recognition

## Installation

1. Main dependencies

   I test the codes with `cudatoolkit==11.3 torch==1.11`.
  
   - CUDA compatible device, NVIDIA driver installed and CUDA available.
   - PyTorch: `>=1.9.0` is required. [Installation guide from PyTorch](https://pytorch.org/get-started/locally/#start-locally)
   - [Kaldi](https://github.com/kaldi-asr/kaldi) **\[optional\]**: used for speech data preparation and some FST-related operations. This is optional for most of the basic functions. Only if you want to conduct [CTC-CRF](egs/TEMPLATE/exp/asr-ctc-crf) training, this is required.
      
      Besides Kaldi, you could use `torchaudio` for feature extraction. See `egs/[task]/local/data.sh` for preparing data with `torchaudio`.

2. Clone and install Transducer packages

   ```bash
   git clone https://github.com/maxwellzh/Transducer-dev.git speech && cd speech
   # Get help message
   ./install.sh -h
   # Install with default configurations
   #./install.sh
   ```

## Get started

To get started with this project, please refer to [TEMPLATE](egs/TEMPLATE/README.md) for tutorial.

## Tutorials

- [TEMPLATE doc](egs/TEMPLATE/README.md): run some template experiments in minutes.
- [Guideline for configuring settings](docs/configure_guide.md)
- [Contribute to this project](docs/contributing.md)
- [Tutorial for CUSIDE](docs/cuside_ch.md) \[Chinese|中文\]: learn to run experiment with [CUSIDE](https://arxiv.org/abs/2203.16758)
- [Some tips about the usage of third party tools](docs/guide_for_third_party_tools.md)
- [Guide to train models on more than 1500 hours of speech data](docs/how_to_prepare_large_dataset_ch.md) \[Chinese|中文\]
