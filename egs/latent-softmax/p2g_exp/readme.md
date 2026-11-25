# LLM-P2G experiments (Qwen3-1.7B)

This folder trains the phoneme-to-grapheme (P2G) stage of the LLM-P2G interface:
a Qwen3-1.7B model that reads a **phoneme sequence produced by the S2P model**
and writes the corresponding text.

The S2P experiments themselves only describe standard (beam-search) decoding in
their `hyper-p.json`. The extra sampling pass that produces P2G training data is
**not** part of those configs; it is documented here and run by hand.

---

## 1. Why the P2G input is sampled, not 1-best

The P2G model never sees the reference phoneme sequence. It sees what the S2P
model emits, including the S2P errors. To avoid fitting a single, arbitrarily
chosen hypothesis, each training utterance is decoded `N` times with ancestral
sampling (`--do_sample`), giving `N` alternative phoneme sequences for the same
audio.

At training time, `K` of those `N` sequences are drawn per step and the loss is

```
-loss = logsumexp(-loss_k)      # k = 1..K   (QwenTrainer.forward)
```

i.e. a **soft minimum over the `K` candidates**: the model is rewarded if *any*
of the sampled hypotheses converts to the correct text, which matches the
"several plausible transcriptions of the same utterance" reading of the samples.

`N` is 16 in the released data (`*_sample16.pkl`); `K` is the `n_samples` field
of the P2G `hyper-p.json` (8 in the released configs).

The dev set is different on purpose: it uses beam-search 1-best
(`*_beam8_best1.pkl`), so the dev curve is deterministic.

---

## 2. Data flow

```
S2P checkpoint
    │
    ├─ cat.ctc.decode_phn --do_sample          →  <prefix> + <prefix>.nbest   (train, N samples/utt)
    └─ cat.ctc.decode_phn  (beam, no sampling)  →  <prefix> + <prefix>.nbest   (dev, 1-best)
                │
                ▼
    local/pkl_Qwen-P2G_data.py --nbest_input
                │
                ▼
    pkl/train.pkl   {"input": [[n phoneme strings]], "label": [text]}
    pkl/dev.pkl     {"input": [[1 phoneme string]],  "label": [text]}
                │
                ▼
    cat.s2p2g.train_qwen_p2g   (train.bin in hyper-p.json)
```

---

## 3. Step 1 — sample phoneme sequences for the training split

Run this **once per S2P system**, after its checkpoint exists. `--config`,
`--resume` and `--tokenizer` must come from the S2P experiment, not from the
P2G experiment.

```bash
cd <CAT_ROOT>/egs/latent-softmax
export PYTHONPATH=<CAT_ROOT>:$PYTHONPATH

python -m cat.ctc.decode_phn \
    --config   mono_exp/librispeech/config.json \
    --resume   mono_exp/librispeech/check/best-10.pt \
    --tokenizer data/dict_librispeech_own_replace/tokenizer.tknz \
    --input_scp data/librispeech/train-clean-100/feats.scp \
    --output_prefix p2g_exp/mono_librispeech/decode/train_sample16 \
    --beam-size 16 \
    --thread-per-woker 2 \
    --do_sample \
    --sample_each_utt \
    --n_samples 16 \
    --gpu
```

Repeat for every training subset (`train-clean-360`, `train-other-500`, ...) with
a distinct `--output_prefix`, or merge the `feats.scp` files first.

Outputs:

| file | content |
|---|---|
| `<output_prefix>` | one line per utterance, `<utt> <1-best phoneme string>` |
| `<output_prefix>.nbest` | pickle `{utt: {0: (score, "phn seq"), ..., N-1: (score, "phn seq")}}` |

Options that matter here:

| option | meaning |
|---|---|
| `--do_sample` | ancestral sampling instead of beam search |
| `--sample_each_utt` | resample until `n_samples` **distinct** sequences are collected |
| `--n_samples 16` | number of sequences stored per utterance |
| `--beam-size` | unused when sampling |
| `--thread-per-woker`, `--gpu` | runtime only |

Without `--sample_each_utt`, `--n_samples` independent draws may contain
duplicates; `--unique_sampling` is the alternative de-duplication path.

---

## 4. Step 2 — beam-search 1-best for the dev split

Same script, sampling switched off:

```bash
python -m cat.ctc.decode_phn \
    --config   mono_exp/librispeech/config.json \
    --resume   mono_exp/librispeech/check/best-10.pt \
    --tokenizer data/dict_librispeech_own_replace/tokenizer.tknz \
    --input_scp data/librispeech/dev-clean/feats.scp \
    --output_prefix p2g_exp/mono_librispeech/decode/dev_beam8 \
    --beam-size 8 \
    --thread-per-woker 2 \
    --gpu
```

Beam search is the default: omitting `--do_sample` (as above) is enough, no
value has to be passed.

---

## 5. Step 3 — pack the pickle

```bash
python local/pkl_Qwen-P2G_data.py \
    p2g_exp/mono_librispeech/pkl/libri_train_960h_sample16_original.pkl \
    --input_files  p2g_exp/mono_librispeech/decode/train_sample16.nbest \
    --output_files data/librispeech/train-clean-100/text \
    --nbest_input
```

* the positional argument is the output `.pkl`;
* `--input_files` is the `.nbest` file (or several) from step 1/2;
* `--output_files` is confusingly named: it takes the **label** (kaldi `text`)
  files, matched to the n-best entries by utterance id;
* `--nbest_input` must be set, otherwise the script expects `scp`-style input.

The script also rewrites IPA tone letters to pinyin diacritics
(`a˥ → ā`, `a˧˥ → á`, ...) and removes the spaces between phonemes, so the string
can be fed to the Qwen tokenizer as plain text. `--DANP` would emit one row per
sample instead of one row per utterance; the released experiments do not use it.

---

## 6. Step 4 — train

`train.bin` in each `p2g_exp/*/hyper-p.json` is already
`cat.s2p2g.train_qwen_p2g`, so training goes through the normal CAT pipeline, the
same way as the S2P experiments.

---

## 7. Configuration reference

`hyper-p.json`

| key | meaning |
|---|---|
| `train.bin` | `cat.s2p2g.train_qwen_p2g` |
| `inference.infer.bin` | `cat.s2p2g.decode_qwen_p2g` |
| `train.option.trset` | training pkl from step 3 (samples) |
| `train.option.devset` | dev pkl from step 2 (beam 1-best) |
| `train.option.n_samples` | `K`, how many of the stored samples are drawn per step; must be **<= N** stored in the pkl |
| `train.option.prompt` | prompt template, `{p}` is replaced by the phoneme string |
| `train.option.gradient_checkpointing` | enable LoRA + gradient checkpointing |
| `tokenizer.file` | Qwen tokenizer pickle (`data/Qwen/Qwen3.tknz`, `Qwen3_fast.tknz`) |

`config.json`

| key | meaning |
|---|---|
| `trainer.model.type` | `Qwen3FromPretrainedModel` |
| `trainer.model.kwargs.pretrained_model_name_or_path` | Qwen3-1.7B-Base directory |
| `trainer.model.kwargs.lora*` | LoRA rank / alpha / dropout / target modules |
| `decoder.s2p_tokenizer` | tokenizer of the S2P model that generated the phonemes |
| `decoder.s2p_model.init_model` | its checkpoint, used at **inference** by `decode_qwen_p2g` |

---

## 8. Which S2P system feeds which P2G experiment

| P2G experiment | S2P checkpoint | S2P tokenizer | training pkl |
|---|---|---|---|
| `mono_librispeech` | `mono_exp/librispeech/check/best-10.pt` | `data/dict_librispeech_own_replace/tokenizer.tknz` | `libri_train_960h_sample16_original.pkl` |
| `mono_aishell` | `mono_exp/aishell/check/best-10.pt` | `data/dict_aishell_Pinyin2IPA_replace/tokenizer.tknz` | `p2g_exp/mono_librispeech/pkl/aishell_train3sp_sample16.pkl` |
| `multi_standard_softmax` | `multi_exp/bilingual_standard_softmax/check/best-10.pt` | `data/dict_multi_replace/tokenizer.tknz` | own `pkl/train.pkl` |
| `multi_latent_softmax` | `multi_exp/bilingual_latent_softmax/check/best-10.pt` | `data/dict_multi_replace/tokenizer_M10_latent_softmax.tknz` | own `pkl/train.pkl` |
| `cs_exp/ASRU19_standard_P2G` | `cs_exp/ASRU19_standard_S2P/check/best-3.pt` | `data/ASRU2019_jnl/tokenizer.tknz` | `asru2019_train_cs_sample16.pkl` |
| `cs_exp/ASRU19_latent_P2G` | `cs_exp/ASRU19_latent_S2P/check/best-3.pt` | `data/ASRU2019_jnl/tokenizer_latent.tknz` | `asru2019_train_cs_sample16.pkl` |

The tokenizer must be the **same** one the S2P model was trained with: the
sampled sequences are id → symbol decoded with it, and P2G consumes the symbols.

---

## 9. Pitfalls

1. `--do_sample`, `--sample_each_utt` and `--unique_sampling` are switches: pass
   the flag to enable, omit it to disable, do not append a value.
2. `n_samples` at training must not exceed the number of samples stored in the
   pkl; `QwenP2GDataset.set_n_samples` raises otherwise.
3. The pkl is keyed by utterance id, so the `.nbest` and the `text` file must
   cover the same ids; missing ids are skipped with a warning.
4. Sampling is stochastic: two runs give different pkls. Keep the seed fixed (or
   keep the generated pkl) if you need an exact reproduction.
