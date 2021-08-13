## CTC-CRF with wordpieces units

---

`CTC-CRF` with `wordpieces` units or wordpieces-based system shares a lot of similarities with that of phone-based system, where the main difference comes from how lexicon is constructed. In the phone-based system, the lexicon defines a mapping rule between word and phonemes, such relationship is obtained either by human experts or `G2P` models. In contrast to phone-based system, the lexicon in wordpiece-based system aims to map word to its grapheme components. In this section, we will introduce how we implement wordpiece-based system under the `CTC-CRF` framework. 

Such introduction will be given at both high-level and low-level. In the high-level, we first train a tokenization model on the training text. With the tokenization model, we i) tokenize words into subword units for acoustic modeling training and ii) restore words from subword units at the stage of decoding. In the low-level, we adopt [SentencePiece](https://github.com/google/sentencepiece) toolkit for the tokenization. Specifically, `spm_train` `spm_encode` are adopted from [fairseq](https://github.com/pytorch/fairseq/tree/master/scripts) for tokenization model training and encoding respectively.

Concretely, we utilize `spm_train` to train a tokenization model. Once model training finishes, we encode word-level transcripts into subword-level transcripts. Normally, the mapping relationship between word and subword units is built by collecting `<word, subword units>` pairs. In our experiments, we found some words demand normalization in order to be handled properly. Since `SentencePiece` provides us with built-in normalization mechanism, thus we utilize this toolkit to normalize text. Accordinglly, we encode the original text with `--output-format=id` flag using `spm_encode` to build `<word, subword ids>` pairs. In other words, we encode words into subword ids rather than plain subword units. In our implementation, `spm_encode` is applied to `wordlist` to build `<word, subword ids>` pairs. Since subword units like `<bos>` and `<eos>` are not considered during AM training, we remove them explicitly from the modeling units set. Please refer to `local/prepare_bpe_dict.sh` for the implementation.

### References 

- Huahuan Zheng, Wenjie Peng, Zhijian Ou and Jinsong Zhang, "Advancing CTC-CRF Based End-to-End Speech Recognition with Wordpieces and Conformers", arXiv:2107.03007.
