# What's New
- 2025.6: Release the code for the INTERSPEECH 2025 paper "LLM-based phoneme-to-grapheme for phoneme-based speech recognition". [Readme](../egs/llm-p2g/README.md) | [Paper](https://arxiv.org/abs/2506.04711)".
- 2024.8: Release the code for “Low-Resourced Speech Recognition for Iu Mien Language via Weakly-Supervised Phoneme-based Multilingual Pre-training” [Readme](../egs/IuMien/README.md) | [Paper](https://arxiv.org/abs/2407.13292)". 
- 2024.6: Release the code for Streaming multi-channel end-to-end (ME2E) ASR. [Readme](./cuside-array.md) | [Paper](https://arxiv.org/abs/2407.09807)
- 2024.6: Release the code for "Whistle: Data-Efficient Multilingual and Crosslingual Speech Recognition via Weakly Phonetic Supervision". [Readme](../egs/cv-lang10/readme.md) | [Paper](https://arxiv.org/abs/2406.02166)
- 2023.5: Release the code for Exploring Energy-based Language Models with Different Architectures and Training Methods for Speech Recognition. [Readme](./energy-based_LM_training.md) | [Paper](https://arxiv.org/abs/2305.12676)
- 2022.11: Release of v3, including:
    - [RNN-Transducer training and decoding implementation](../egs/TEMPLATE/exp/asr-rnnt) (Huahuan Zheng).
    - [Language model (NN and n-gram) training and inference support](../egs/TEMPLATE/README.md#language-model) (Huahuan Zheng).
    - LM fusion support for ASR, including [Low Order Density Ratio (LODR)](https://arxiv.org/abs/2203.16776) for language model integration (Huahuan Zheng).
    - CUSIDE implementation for training unified streaming / non-streaming models (Keyu An, Huahuan Zheng and Ziwei Li). [Paper](https://arxiv.org/abs/2203.16758) | [Readme](./cuside_ch.md) | [中文说明](./cuside_ch.md)
    - Guide to train models on more than 1500 hours of speech data: [English](./how_to_prepare_large_dataset.md) | [中文说明](./how_to_prepare_large_dataset_ch.md)

- 2022.05: Release the code for Join Acoustics and Phonology (JoinAP) for Multi/Cross-lingual ASR. [Readme](joinap.md) | [Tutorial](joinap_tutorial_ch.md) | [ASRU2021 Paper](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ASRU21_JoinAP.pdf) | [Slides](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/asru2021_JoinAP_slides.pdf) | [Video](https://www.bilibili.com/video/BV1X44y1Y7zm)

- 2021.07: add support of Deformable TDNN by Keyu An. [INTERSPEECH2021 Paper](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/is2021_deformable.pdf)

- 2021.07: add support of Wordpieces by Wenjie Peng. [Readme](wordpieces.md) | [Paper](https://arxiv.org/abs/2107.03007)

- 2021.05: add support of Conformer and SpecAug by Huahuan Zheng. [Readme](conformer.md) | [Paper](https://arxiv.org/abs/2107.03007)
