# Tatar
Author: Ma, Te (mate153125@gmail.com)
## 1. Text normalization 

(1) Before genetaring lexicon, we need to normalize text. The code of text normalization for __Tatar__ is in the script named [`text_norm.sh`](text_norm.sh).

## 2. Lexicon generation and correction

We use the FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit, [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), to create the pronunciation lexicon. The trained FSTs for use with Phonetisaurus is provided in [LanguageNet](https://github.com/uiuc-sst/g2ps#languagenet-grapheme-to-phoneme-transducers).

Note that the above G2P procedure is not perfect. As noted in `LanguageNet`, "PERs range from 7% to 45%".
The G2P-generated lexicon needs to be corrected. The correction step is based on [the LanguageNet symbol table for __Tatar__](https://github.com/uiuc-sst/g2ps/blob/masterTatar/Tatar_wikipedia_symboltable.html). The code of this step of lexicon correction is in the script named [`lexicon.sh`](lexicon.sh).

(1) For the phoneme `/t͡ʃ/`, we need to revise and `/tʃ/` to a single phoneme `/t͡ʃ/`. So as for `/d͡ʒ/`.

| Phoneme from G2P | Phoneme corrected |
| ------ | ------ |
| `tʃ` | `t͡ʃ` |
| `dʒ` | `d͡ʒ` |

The generated lexicon from the G2P procedure is named [`lexicon_tt.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tt/lexicon_tt.txt). The set of IPA phonemes appeared in the lexicon is saved in [`phone_list.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tt/phone_list.txt).

### Note：
Word pronunciations of the language were not available, so check of phonemes were not performed.