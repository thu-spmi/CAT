
# 1. Overview

This directory contains the experimental codes corresponding to the research on "Phoneme-based Cross-lingual ASR Training without Pronunciation Dictionary via Joint Stochastic Approximation". The code is designed to reproduce the experiments described in the paper, covering data preprocessing, model training, and evaluation steps.

# 2. Results

## 2.1 SPG-JSA Crosslingual ASR Experiments

**Settings**: 
Two languages (Polish and Indonesian) from the Common Voice dataset were used. Only 10 minutes of phoneme - annotated data per language was utilized. Two schemes, "SPG init from G2P" and SPG - JSA, were compared.

**Results**:  
The following result table shows that PERs (\%) and WERs (\%) for SPG-JSA experiment on Common Voice dataset. FT denotes fine-tuning. MLS denotes marginal likelihood scoring. NA denotes not applied.
> Tips:  $^\dagger$ denotes results from whistle paper. More experiment details see [cv-lang10](../cv-lang10).

|                Exp.                | **Polish** |        |      |      | **Indonesian** |        |       |          |
| :--------------------------------: | :--------: | :----: | ---- | ---- | -------------- | ------ | ----- | -------- |
|                                    |    PER     | w/o LM | w LM | MLS  | PER            | w/o LM | w LM  | MLS      |
| **Monolingual phoneme $^\dagger$** |    2.82    |   NA   | 4.97 | NA   | 5.74           | NA     | 3.28  | NA       |
| **Monolingual subword $^\dagger$** |     NA     | 19.38  | 7.12 | NA   | NA             | 31.96  | 10.85 | NA       |
| **Whistle phoneme FT $^\dagger$**  |    1.97    |   NA   | 4.30 | NA   | 4.79           | NA     | 2.43  | NA       |
| **Whistle subword FT $^\dagger$**  |     NA     |  5.84  | 3.82 | NA   | NA             | 12.48  | 2.92  | NA       |
|       **SPG init from G2P**        |   17.72    |  8.73  | 4.68 | 5.91 | 21.85          | 10.15  | 3.81  | 3.09     |
|    **+ P2G augmentation**     |   17.72    |  5.93  | 4.97 | 5.88 | 21.85          | 6.34   | 3.44  | 2.91     |
|            **SPG-JSA**             |   17.35    |  8.19  | 4.65 | [3.93](./jsa_exp/SPG-JSA_polish_semi-supervised_100utts/)  | 20.66          | 9.04   | 3.26  | [2.47](./jsa_exp/SPG-JSA_indonesian_semi-supervised_100utts/)   |
|    **+ P2G augmentation**     |   17.35    |  4.64  | 4.37 | **[3.64](./jsa_exp/SPG-JSA_polish_semi-supervised_100utts/)**  | 20.66          | 4.55   | 2.92  | **[2.31](./jsa_exp/SPG-JSA_indonesian_semi-supervised_100utts/)** |

## 2.2 Language Domain Adaptation Experiments

**Settings**: 
The SPG-JSA model trained on Common Voice was tested on target domain test set, and dLanguage Domain Adaptation was further applied.

**Results**:  

|           **Exp.**           | **Polish** |       |           | **Indonesian** |       |           |
| :--------------------------: | :--------: | :---: | --------- | -------------- | ----- | --------- |
|                              |   w/o LM   | w LM  | MLS       | w/o LM         | w LM  | MLS       |
| **Whistle subword FT on CV** |   33.46    | 22.58 | NA        | 43.69          | 12.39 | NA        |
|      **SPG-JSA on CV**       |   35.18    | 29.04 | 26.79     | 39.19          | 16.93 | 14.28     |
|   **+ LDA training**    |   28.87    | 23.84 | **20.57** | 30.69          | 12.68 | **11.23** |

## 2.3 Effect of Supervised Data Amount

| **Amount of supervised data**        | **Polish** |          |          |          | **Indonesian** |          |          |          |
| ------------------------------------ | ---------- | -------- | -------- | -------- | -------------- | -------- | -------- | -------- |
|                                      | PER        | w/o LM   | w LM     | MLS      | PER            | w/o LM   | w LM     | MLS      |
| **unsupervised**                     | 50.24      | 13.43    | 6.20     | 5.05     | 32.71          | 11.09    | 3.74     | 2.80     |
| **+ P2G augmentation**          | 50.24      | 6.08     | 5.28     | 4.48     | 32.71          | 5.33     | 3.28     | 2.47     |
| **20 sentences (about 2 minutes)**   | 27.35      | 8.25     | 5.17     | 4.25     | 27.37          | 9.01     | 3.39     | 2.66     |
| **+ P2G augmentation**          | 27.35      | 5.31     | 4.78     | 3.96     | 27.37          | 5.45     | 3.04     | 2.47     |
| **100 sentences (about 10 minutes)** | **17.35**  | 8.19     | 4.65     | 3.93     | **20.66**      | 9.04     | 3.26     | 2.47     |
| **+ P2G augmentation**          | **17.35**  | **4.64** | **4.37** | **3.64** | **20.66**      | **4.55** | **2.92** | **2.31** |

# 3. Code Structure

The code is organized into several main directories:
- **`jsa_exp`**: includes experiment directories for training SPG-JSA models. The experimental process of SPG-JSA is detailed here. **You should start your experiment from here**.
- **`g2p_exp`**: includes experiment directories for training G2P models.
- **`p2p_exp`**: includes experiment directories for training P2G models.
- **`s2p_exp`**: includes experiment directories for training S2P models.
- **`data`**: contains meta data, speech features, LM and FST graph files.
- **`local`**: contains some trial scripts.
- **`cat` and `utils`**: include codes for ASR training and evaluation in CAT.

# 4. How to Use
The detailed training steps of SPG-JSA start from [here](./jsa_exp/readme.md). 
1. **Data Preparation**
    - Download the necessary datasets, which can be the Common Voice, VoxPopuli, or your custom dataset.
    - Format the data in the CAT style. You can refer to the [Common Voice repository](../commonvoice) for guidance.
    - Update the data path in the [`metainfo.json`](./data/metainfo.json) file.
2. **Model Training**
    - Refer to the instructions provided in the experiment directory (the link is available in the result table).
    - Proceed to train your own model accordingly.
3. **Evaluation without Language Model (LM)**
    - To evaluate the Phoneme Error Rate (PER) and Word Error Rate (WER) without an LM, follow the instructions given in the experiment directory (linked in the result table).
4. **Evaluation with Language Model (LM)**
    - Prepare the N - gram Language Model (LM) and the Finite State Transducer (FST) Graph. You can refer to the [`lm_decode.sh`](./local/lm_decode.sh) script for details. These are required for evaluating the WER with an LM and performing DLS decoding.
    - Then, follow the instructions in the corresponding experiment directory for the evaluation process.
