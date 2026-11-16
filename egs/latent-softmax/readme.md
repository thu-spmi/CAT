# 1. Overview

This directory contains the experimental codes corresponding to the research on "Latent Softmax for Data-Efficient Phoneme-Based Multilingual ASR Across Tonal and Non-Tonal Languages". The code is designed to reproduce the experiments described in the paper, covering phoneme dictionary preparation, speech-to-phoneme (S2P) training, LLM-based phoneme-to-grapheme (LLM-P2G) conversion, projector-based ASR, and Mandarin-English code-switching adaptation.

Latent Softmax is used for multilingual S2P training with both tonal and non-tonal languages. Tone-marked vowels are modeled as subclasses, while non-tonal base vowels are treated as major classes whose compatible subclasses are marginalized in the CTC loss. This keeps tone distinctions for Mandarin and still allows English base-vowel observations to share training evidence with the tone-marked vowel subclass space.

# 2. Results

## 2.1 Main Experiments

**Settings**: 
AISHELL-1 Mandarin is used as the tonal-language benchmark, and LibriSpeech English is used as the non-tonal-language benchmark. AISHELL-1 reports tonal PER and CER/WER on the test set. LibriSpeech reports non-tonal PER and WER on `test-clean` and `test-other`. The S2P encoder follows a Conformer-based configuration comparable to Whistle-small. Downstream word recognition is evaluated with two interfaces: Qwen3-1.7B based LLM-P2G and a projector-based S2P-to-LLM interface. The projector interface is trained in two stages: first train the projector with the S2P encoder and LLM frozen, then train the LLM LoRA parameters with the S2P encoder and projector fixed.

**Results**:  
The following table shows tonal PERs (%), non-tonal PERs (%), and WERs/CERs (%) on AISHELL-1 and LibriSpeech. AISHELL-1 uses tonal PER for S2P and CER for the final Chinese transcription. LibriSpeech uses non-tonal PER and WER. The PER of the same S2P model is shared by its downstream LLM-P2G and projector rows.

| Exp. | **AISHELL-1** |  | **LibriSpeech test-clean** |  | **LibriSpeech test-other** |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Tonal PER | CER | Non-tonal PER | WER | Non-tonal PER | WER |
| **Monolingual phoneme S2P** | [1.66](./mono_exp/aishell/) |  | [1.20](./mono_exp/librispeech/) |  | [3.87](./mono_exp/librispeech/) |  |
| **+ LLM-P2G** |  | [5.40](./p2g_exp/mono_aishell/) |  | [4.12](./p2g_exp/mono_librispeech/) |  | [9.47](./p2g_exp/mono_librispeech/) |
| **+ projector** |  | [5.27](./projector_exp/06_O22-aishell-Pr2/) |  | [3.29](./projector_exp/08_O21-librispeech-Pr2/) |  | [8.24](./projector_exp/08_O21-librispeech-Pr2/) |
| **[Standard Softmax S2P](./multi_exp/bilingual_standard_softmax/)** | 2.03 |  | 1.60 |  | 5.00 |  |
| **[+ LLM-P2G](./p2g_exp/multi_standard_softmax/)** |  | 5.63 |  | 4.29 |  | 10.70 |
| **[+ projector](./projector_exp/02_M19-mono-Pr2/)** |  | 5.61 |  | 3.92 |  | 9.61 |
| **[Latent Softmax S2P](./multi_exp/bilingual_latent_softmax/)** | 1.86 |  | 1.32 |  | 4.37 |  |
| **[+ LLM-P2G](./p2g_exp/multi_latent_softmax/)** |  | 5.26 |  | 3.91 |  | 9.89 |
| **[+ projector](./projector_exp/10_M21-multi-Pr2/)** |  | 5.52 |  | 3.63 |  | 9.14 |

## 2.2 Code-switching Experiments

**Settings**: 
Mandarin-English code-switching adaptation is evaluated on ASRU2019 and CS-Dialogue. ASRU2019 uses the 200-hour `Train_CS` subset for adaptation, the 40-hour `Dev_CS` subset as the development set, and reports results on `Test_CS`. The extra Mandarin-only training data in ASRU2019 is not used in these experiments. CS-Dialogue uses the official speaker-independent train/dev/test splits. All code-switching systems are initialized from the corresponding main-experiment systems. For the projector interface, the S2P model is fine-tuned first and then the LLM LoRA parameters are adapted while the projector is kept frozen.

**Results**:  
The following table shows non-tonal PERs (%) and mixed error rates (MERs, %) on Mandarin-English code-switching datasets. Non-tonal PER is measured after removing tone marks from the S2P output and reference. MER is measured on the final Mandarin-character and English-word mixed token sequence.

| Exp. | **ASRU2019** |  | **CS-Dialogue** |  |
| :--- | :---: | :---: | :---: | :---: |
|  | Non-tonal PER | MER | Non-tonal PER | MER |
| **Standard Softmax S2P** | [15.97](./cs_exp/ASRU19_standard_S2P/) |  | [12.28](./cs_exp/CS_Dialogue_standard_S2P/) |  |
| **+ LLM-P2G** |  | [13.69](./cs_exp/ASRU19_standard_P2G/) |  | - |
| **+ projector** |  | [14.08](./projector_exp/11_C5-asru19-Pr2_init_from_projector02/) |  | [15.53](./projector_exp/18_C13-cs-dialog-Pr2_init_from_projector02/) |
| **Latent Softmax S2P** | [16.59](./cs_exp/ASRU19_latent_S2P/) |  | [11.67](./cs_exp/CS_Dialogue_latent_S2P/) |  |
| **+ LLM-P2G** |  | [13.83](./cs_exp/ASRU19_latent_P2G/) |  | - |
| **+ projector** |  | [13.71](./projector_exp/14_C5-asru19-Pr2_init_from_projector10/) |  | [14.06](./projector_exp/19_C12-cs-dialog-Pr2_init_from_projector10/) |

# 3. Code Structure

The code is organized into several main directories:

- **`mono_exp`**: includes monolingual AISHELL-1 and LibriSpeech S2P baseline experiments.
- **`multi_exp`**: includes multilingual S2P experiments. `bilingual_standard_softmax` is the standard softmax baseline, and `bilingual_latent_softmax` is the Latent Softmax system. **You should start the main S2P comparison from here**.
- **`p2g_exp`**: includes Qwen3-1.7B LLM-P2G experiments. These experiments take sampled S2P phoneme sequences as input and train the text generation stage. See [`p2g_exp/readme.md`](./p2g_exp/readme.md) for how those sequences are generated and packed.
- **`projector_exp`**: includes projector-based ASR experiments. `Pr1` directories train the projector, and `Pr2` directories train the LLM LoRA parameters initialized from the corresponding `Pr1` checkpoint.
- **`cs_exp`**: includes Mandarin-English code-switching S2P and LLM-P2G adaptation experiments.
- **`local`**: contains helper scripts for lexicon construction, OOV detection, major-class to subclass mapping, and Qwen-P2G data packing.
- **`src/latent_ctc`** (outside this recipe directory): the GPU Latent CTC loss used by the latent-softmax experiments. See [`src/latent_ctc/readme.md`](../../src/latent_ctc/readme.md) for how to build and install it.
- **`Chinese_and_English_phone_set.md`**: documents the Chinese and English IPA phone sets, tone-marked vowels, and phone replacement rules used in the experiments.

Each experiment directory contains:

- **`hyper-p.json`**: CAT pipeline configuration, including dataset names, tokenizer path, training entry, inference entry, and GPU environment.
- **`config.json`**: model configuration, including encoder, decoder, Latent Softmax mapping, Qwen model path, projector settings, and scheduler settings.
- **`readme.md`**: experiment-specific training information, hardware information, training curve, and final evaluation result.
- **`monitor.png`**: TensorBoard-style training curve exported for quick inspection.

# 4. How to Use

The detailed configurations are stored in the linked experiment directories in the result tables. A typical reproduction flow is:

1. **Prepare the CAT environment**
    - Run the CAT pipeline from the recipe working directory that contains `data/metainfo.json`. For this recipe, this is normally `<CAT_ROOT>/egs/latent-softmax` after data preparation, where `<CAT_ROOT>` is the root of your checkout.
    - Add the CAT project root to `PYTHONPATH` when using the shared top-level CAT code.
    - Make sure the CAT modules referenced by `hyper-p.json` are available in your checkout. The exact historical code version for each experiment is recorded in the `commit` field of `hyper-p.json`.

2. **Prepare data and tokenizers**
    - Prepare AISHELL-1, LibriSpeech, ASRU2019, and CS-Dialogue in CAT format.
    - Each dataset entry in `hyper-p.json` should exist in `data/metainfo.json` under the recipe working directory, and the corresponding `feats.scp` and `text` files should be available.
    - Prepare the lexicons and tokenizers referenced by `hyper-p.json`, such as `data/dict_multi_replace/tokenizer_M10_latent_softmax.tknz`, `data/ASRU2019_jnl/tokenizer_latent.tknz`, and `data/Qwen/Qwen3.tknz`.
    - For Mandarin lexicon construction and phone normalization, refer to [`Chinese_and_English_phone_set.md`](./Chinese_and_English_phone_set.md) and the scripts in [`local`](./local).

3. **Train and evaluate S2P**
    - For monolingual baselines, use the experiment directories in [`mono_exp`](./mono_exp).
    - For the standard softmax multilingual baseline, use [`multi_exp/bilingual_standard_softmax`](./multi_exp/bilingual_standard_softmax/).
    - For Latent Softmax, use [`multi_exp/bilingual_latent_softmax`](./multi_exp/bilingual_latent_softmax/).
    - The CAT pipeline stages are controlled by `hyper-p.json`. For example:

```bash
cd <CAT_ROOT>/egs/latent-softmax
PYTHONPATH=../.. python ../../cat/utils/pipeline/asr.py multi_exp/bilingual_latent_softmax --start_stage 1 --stop_stage 4
```

4. **Train LLM-P2G**
    - Use the directories in [`p2g_exp`](./p2g_exp). The `config.json` files define the Qwen3-1.7B model and the S2P checkpoint used to generate phoneme input.
    - The phoneme input is produced by an explicit sampling pass over the S2P model, then packed into a pickle; both steps are described in [`p2g_exp/readme.md`](./p2g_exp/readme.md). The S2P `hyper-p.json` files only contain standard (beam-search) decoding.
    - The LLM-P2G experiments are trained after the corresponding S2P model is available.

5. **Train projector-based ASR**
    - Use the directories in [`projector_exp`](./projector_exp).
    - Run the `Pr1` experiment first to train the projector.
    - Run the corresponding `Pr2` experiment next to train the LLM LoRA parameters. The `Pr2` `hyper-p.json` points to the `Pr1` checkpoint through `train:option:init_model`.

6. **Run code-switching adaptation**
    - Use the directories in [`cs_exp`](./cs_exp) for S2P and LLM-P2G adaptation.
    - Use [`projector_exp/11_C5-asru19-Pr2_init_from_projector02`](./projector_exp/11_C5-asru19-Pr2_init_from_projector02/) and [`projector_exp/14_C5-asru19-Pr2_init_from_projector10`](./projector_exp/14_C5-asru19-Pr2_init_from_projector10/) for ASRU2019 projector adaptation.
    - For CS-Dialogue projector results, use the corresponding initialized projector/LoRA adaptation configuration following the same `Pr1` to `Pr2` protocol.

7. **Check results**
    - The summarized numbers are shown in the tables above.
    - More detailed logs, checkpoint averaging settings, and training curves are recorded in the `readme.md`, `hyper-p.json`, `config.json`, and `monitor.png` files inside each experiment directory.

# 5. Dependencies

Install the CAT requirements first (`pip install -r requirements.txt`, plus the editable `-e` entries). The recipe additionally needs:

| Dependency | Needed by | Notes |
| :--- | :--- | :--- |
| `ctcdecode` (parlance, with KenLM) | every decoding entry, and the dev-set error-rate evaluation | not on PyPI as a wheel; build from source |
| `ctc_align` | `--do_sample` in `cat.ctc.decode_phn` / `cat.ctc.decode_latent_softmax` | already listed in `requirements.txt` |
| `peft` | `Qwen3FromPretrainedModel` (`p2g_exp`, `cs_exp/ASRU19_standard_P2G`, `cs_exp/ASRU19_latent_P2G`) | `pip install peft` |
| `latent_ctc` | `cs_exp/CS_Dialogue_latent_S2P`, and the default loss of `cat.ctc.train_latent_softmax` | shipped in this repository under `src/latent_ctc`; build it with `make`; see [`src/latent_ctc/readme.md`](../../src/latent_ctc/readme.md) |

`latent_ctc` is the GPU Latent CTC loss: the network keeps the subclass vocabulary, and the subclasses of a base-vowel major class are marginalized inside the CTC forward-backward. It replaces the earlier `ctc_crf.LATENT_CTC_LOSS` entry point, so the call site is now `latent_ctc.LATENT_CTC_LOSS`. The package is a trimmed copy of `src/ctc_crf` with the CTC-CRF denominator and its OpenFST dependency removed, and has only been validated on NVIDIA H100, L40S, RTX 3090 and RTX 4090. The criterion is resolved lazily at the first training step, so decoding existing checkpoints works without the extension.
