# Kirghiz
Author: Ma, Te (mate153125@gmail.com)
## 1. Text normalization 

(1) Before genetaring lexicon, we need to normalize text. The code of text normalization for __Kirghiz__ is in the script named [`text_norm.sh`](text_norm.sh).

## 2. Lexicon generation and correction

We use the FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit, [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), to create the pronunciation lexicon. The trained FSTs for use with Phonetisaurus is provided in [LanguageNet](https://github.com/uiuc-sst/g2ps#languagenet-grapheme-to-phoneme-transducers).

Note that the above G2P procedure is not perfect. As noted in `LanguageNet`, "PERs range from 7% to 45%".
The G2P-generated lexicon needs to be corrected. The correction step is based on [the LanguageNet symbol table for __Kirghiz__](https://github.com/uiuc-sst/g2ps/blob/masterKirghiz/Kirghiz_Cyrillic_wikipedia_symboltable.html). The code of this step of lexicon correction is in the script named [`lexicon.sh`](lexicon.sh).

(1) We remove some special symbols such as accent symbols to enable sharing more phonemes between different languages.

| Removed symbols | Note |
| ------ | ------ |
| `ː` | Accent | 
| `ˈ` | Long vowel |
| `ˌ` | Syllable |

The generated lexicon from the G2P procedure is named [`lexicon_ky.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ky/lexicon_ky.txt). The set of IPA phonemes appeared in the lexicon is saved in [`phone_list.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ky/phone_list.txt).

### Note：
Word pronunciations of the language were not available, so check of phonemes were not performed.