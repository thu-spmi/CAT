# Language process
## 1. Text normalization 
The training dataset of our models are sourced from the publicly available [`Common Voice`](https://commonvoice.mozilla.org/) 11.0. There are some redundant symbols or alien words which may affect model performance, so we do text normalize to remove them for each language.

## 2. Lexicon generation and correction
The %PER of FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit that we used to generate pronunciation lexicon range from 7% to 45%, so the lexicon need to be corrected.

## Check of phonemes
After lexicon correction, the final lexicon is also not perfect, with some noise. We further check our phonemes by referring to the IPA symbol table in LanguageNet and Phoible, with Google Translate listening test.