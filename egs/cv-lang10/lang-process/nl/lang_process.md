# Dutch
Author: Ma, Te (mate153125@gmail.com)
## 1. Text normalization 

(1) The G2P models cannot recognize alien words, so we choose to remove sentences that containing alien words. They are listed in the file [`Dutch_alien_sentences.txt`](/home/mate/cat_multilingual/egs/cv-lang10/data-process/nl/Dutch_alien_sentences.txt)

(2) Before creating lexicon, we need to normalize text. The code of text normalization for __Dutch__ is in the script named [`text_norm.sh`](./text_norm.sh).

## 2. Lexicon generation and correction

We use the FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit, [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), to create the pronunciation lexicon. The trained FSTs for use with Phonetisaurus is provided in [LanguageNet](https://github.com/uiuc-sst/g2ps#languagenet-grapheme-to-phoneme-transducers).

Note that the above G2P procedure is not perfect. As noted in `LanguageNet`, "PERs range from 7% to 45%".
The G2P-generated lexicon needs to be corrected. The correction step is based on [the LanguageNet symbol table for __Dutch__](https://github.com/uiuc-sst/g2ps/blob/masterDutch/Dutch_wikipedia_symboltable.html). The code of this step of lexicon correction is in the script named [`lexicon.sh`](./lexicon.sh).

(1) We remove some special symbols such as accent symbols and long vowel symbols to enable sharing more phonemes between different languages. 
    
| Removed symbols | Note |
| ------ | ------ |
| `ː` | Accent | 
| `ˈ` | Long vowel |
| `ʲ` | Velarization |

(2) For the phoneme `/d͡ʒ/`, we need to revise `/dʒ/` to a single phoneme `/d͡ʒ/`. For phonemes `/œɪ/` and `/ɛɪ/`, we should split them to `/œ ɪ/` and `/ɛ ɪ/`

| Phoneme from G2P | phonemes corrected |
| ------ | ------ |
| `dʒ` | `d͡ʒ` |
| `œɪ` | `œ y` |
| `ɛɪ` | `ɛ i` |

## 3. Check of phonemes

Strictly speaking, one phoneme might correspond to multiple phones (those phones are referred to as the allophones). Note that our above procedure removes the diacritic, the notion of phonemes in this work is a looser one.

The generated lexicon from the G2P procedure is named [`lexicon_nl.txt`](/home/mate/cat_multilingual/egs/cv-lang10/dict/nl/lexicon_nl.txt). The set of IPA phonemes appeared in the lexicon is saved in [`phone_list.txt`](/home/mate/cat_multilingual/egs/cv-lang10/dict/nl/phone_list.txt). We further check `phone_list.txt`, by referring to the following two phoneme lists and with listening tests.  

* IPA symbol table in LanguageNet, which, thought by LanguageNet, contains all the phones in the language:
https://github.com/uiuc-sst/g2ps/blob/masterDutch/Dutch_wikipedia_symboltable.html
  
* IPA symbol table in Phoible: 
https://phoible.org/languages/dutc1256. For each language, there may exist multiple phoneme inventories, which are archived at the Phoible website. 
For __Dutch__, we choose the first one as the main reference for phoneme checking, which is [PH 1050](https://phoible.org/inventories/view/1050).

Note that the G2P procedure is not perfect,  the G2P-generated `phone_list.txt` is not exactly the same as the ideal IPA symbol table in LanguageNet. Further, the IPA symbol table in LanguageNet may also differ from other IPA symbol tables from other linguistic resources (e.g., Phoible). So we need to check. The inconsistencies are recorded in the following. The lexicon is not modified, since a complete modification of the whole lexicon requires non-trivial manual labor. The final lexicon is not perfect, with some noise.

### Checking process

For each IPA phoneme in  `phone_list.txt`, its sound obtained from Wikipedia is listened. 
A word, which consists of this IPA phoneme, is arbitrarily chosen from the lexicon and listened from Google Translate.
By comparing these two sounds, we could do phoneme check, which is detailed as follows.

#### Check whether there is any inconsistency between `phone_list.txt`, IPA symbol table in LanguageNet, and IPA symbol table in Phoible
A phoneme in `phone_list.txt` should appear in both the IPA symbol table in LanguageNet G2P and the IPA symbol tables in Phoible.

#### Check whether the G2P labeling is correct
The Wikipedia sound of the phoneme should match that appeared in the corresponding position in the Google Translate pronunciation of the word, which consists of this IPA phoneme.

If either of the above two checks fail, it means that the lexicon contains some errors and needs to be further corrected.

### Checking result

The checking result is shown in the following table. Clicking the hyperlinks will download the sound files for your listening.
* The first column shows the phonemes in `phone_list.txt`.  
* The second and third columns show the word and its G2P labeling. The word's G2P labeling consists of the phoneme in the first column.
* The last column contains some checking remarks.

| IPA symbol in `phone_list.txt` | Word |  <div style="width: 205pt">G2P labeling result | Note |
| ------ | ------ | ------ | ------ |
| [`ɣ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɣ.mp3) | [vrachtwa`g`enchauffeurs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/vrachtwagenchauffeurs.mp3) | v r ɑ x t ʋ a `ɣ` ə n ʃ o f ø r s |  |
| [`ʃ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʃ.mp3) | [vrachtwagen`ch`auffeurs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/vrachtwagenchauffeurs.mp3) | v r ɑ x t ʋ a ɣ ə n `ʃ` o f ø r s | The phoneme `/ʃ/` is not contained in phoneme table of Phoible, but it sounds correct |
| [`ɡ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɡ.mp3) | [jun`g`le](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/jungle.mp3) | d͡ʒ ʉ ŋ `ɡ` ə l | The phoneme `/ɡ/` is not contained in any phoneme tables of LanguageNet or Phoible, but sounds correct because the word is a English word |
| [`ɯ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɯ.mp3) | [f`ou`tenmarge](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/foutenmarge.mp3) | f `ɯ` t ə m ɑ r ʒ ə | Incorrect G2P labeling. The phoneme `/ɯ/` is not contained in any phoneme tables of LanguageNet or Phoible, and needs to be corrected to `/ɔ u/` |
| [`ʉ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʉ.mp3) | [j`u`ngle](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/jungle.mp3) | d͡ʒ `ʉ` ŋ ɡ ə l | Incorrect G2P labeling. The phoneme `/ʉ/` is not contained in any phoneme tables of Phoible, and needs to be corrected to `/u/` |
| [`ɒ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɒ.mp3) | [alcoholcontr`o`le](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/alcoholcontrole.mp3) | ɑ l k o h ɔ l k ɔ n t r `ɒ` l ə | The phoneme `/ɒ/` is not contained in [PH 1050](https://phoible.org/inventories/view/1050), but contained in [UZ 2174](https://phoible.org/inventories/view/2174) |
| [`ʋ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʋ.mp3) | [`v`rachtwagenchauffeurs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/vrachtwagenchauffeurs.mp3) | v r ɑ x t `ʋ` a ɣ ə n ʃ o f ø r s | Incorrect G2P labeling. The phoneme `/ʋ/` is not contained in any phoneme tables of Phoible, and needs to be corrected to `/v/` |
| [`ɑ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɑ.mp3) | [vr`a`chtwagenchauffeurs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/vrachtwagenchauffeurs.mp3) | v r `ɑ` x t ʋ a ɣ ə n ʃ o f ø r s |  |
| [`ɪ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɪ.mp3) | [afkoel`i`ngsperiode](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/afkoelingsperiode.mp3) | ɑ f k u l `ɪ` ŋ s p e r i j o d ə |  |
| [`a`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/a.mp3) | [vrachtw`a`genchauffeurs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/vrachtwagenchauffeurs.mp3) | v r ɑ x t ʋ `a` ɣ ə n ʃ o f ø r s |  |
| [`b`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/b.mp3) | [meerderheids`b`esluitvorming](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/meerderheidsbesluitvorming.mp3) | m e r d ə r h ɛ i d z `b` ə s l œ y t f ɔ r m ɪ ŋ |  |
| [`d`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/d.mp3) | [afkoelingsperio`d`e](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/afkoelingsperiode.mp3) | ɑ f k u l ɪ ŋ s p e r i j o `d` ə |  |
| [`dʒ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/dʒ.mp3) | [`j`ungle](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/jungle.mp3) | `d͡ʒ` ʉ ŋ ɡ ə l | The phoneme `/dʒ/` is not contained in any phoneme tables of LanguageNet or Phoible, but sounds correct. And the same pronunciation for `/dʒ/` and `/d͡ʒ/`, so replacing `/dʒ/` with `/d͡ʒ/` |
| [`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/e.mp3) | [afkoelingsp`e`riode](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/afkoelingsperiode.mp3) | ɑ f k u l ɪ ŋ s p `e` r i j o d ə |  |
| [`ə`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ə.mp3) | [foutenmarg`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/foutenmarge.mp3) | f ɯ t ə m ɑ r ʒ `ə` | The phoneme `/ə/` is not contained in [PH 1050](https://phoible.org/inventories/view/1050), but contained in [UZ 2170](https://phoible.org/inventories/view/2170) |
| [`ɛ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɛ.mp3) | [meerderh`e`idsbesluitvorming](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/meerderheidsbesluitvorming.mp3) | m e r d ə r h `ɛ` i d z b ə s l œ y t f ɔ r m ɪ ŋ |  |
| [`f`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/f.mp3) | [`f`outenmarge](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/foutenmarge.mp3) | `f` ɯ t ə m ɑ r ʒ ə |  |
| [`h`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/h.mp3) | [alco`h`olcontrole](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/alcoholcontrole.mp3) | ɑ l k o `h` ɔ l k ɔ n t r ɒ l ə | The phoneme `/h/` is not contained in [PH 1050](https://phoible.org/inventories/view/1050), but contained in [UZ 2170](https://phoible.org/inventories/view/2170) |
| [`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/i.mp3) | [afkoelingsper`i`ode](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/afkoelingsperiode.mp3) | ɑ f k u l ɪ ŋ s p e r `i` j o d ə |  |
| [`j`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/j.mp3) | [afkoelingsperiode](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/afkoelingsperiode.mp3) | ɑ f k u l ɪ ŋ s p e r i `j` o d ə | Incorrect G2P labeling. The redundant phonemes `/j/` should be removed |
| [`k`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/k.mp3) | [af`k`oelingsperiode](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/afkoelingsperiode.mp3) | ɑ f `k` u l ɪ ŋ s p e r i j o d ə |  |
| [`l`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/l.mp3) | [afkoe`l`ingsperiode](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/afkoelingsperiode.mp3) | ɑ f k u `l` ɪ ŋ s p e r i j o d ə |  |
| [`m`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/m.mp3) | [fouten`m`arge](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/foutenmarge.mp3) | f ɯ t ə `m` ɑ r ʒ ə |  |
| [`n`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/n.mp3) | [alcoholco`n`trole](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/alcoholcontrole.mp3) | ɑ l k o h ɔ l k ɔ `n` t r ɒ l ə |  |
| [`ŋ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ŋ.mp3) | [ju`n`gle](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/jungle.mp3) | d͡ʒ ʉ `ŋ` ɡ ə l |  |
| [`o`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/o.mp3) | [alc`o`holcontrole](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/alcoholcontrole.mp3) | ɑ l k o h ɔ l k ɔ n t r ɒ l ə |  |
| [`ø`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ø.mp3) | [vrachtwagenchauff`eu`rs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/vrachtwagenchauffeurs.mp3) | v r ɑ x t ʋ a ɣ ə n ʃ o f `ø` r s |  |
| [`ɔ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɔ.mp3) | [alcoh`o`lcontrole](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/alcoholcontrole.mp3) | ɑ l k o h `ɔ` l k ɔ n t r ɒ l ə |  |
| [`œ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/œ.mp3) | [meerderheidsbesl`u`itvorming](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/meerderheidsbesluitvorming.mp3) | m e r d ə r h ɛ i d z b ə s l `œ` y t f ɔ r m ɪ ŋ | The letter `ui` is pronounced as `/œ y/` |
| [`p`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/p.mp3) | [afkoelings`p`eriode](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/afkoelingsperiode.mp3) | ɑ f k u l ɪ ŋ s `p` e r i j o d ə |  |
| [`r`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/r.mp3) | [afkoelingspe`r`iode](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/afkoelingsperiode.mp3) | ɑ f k u l ɪ ŋ s p e `r` i j o d ə | The phoneme `/r/` is not contained in [PH 1050](https://phoible.org/inventories/view/1050), but contained in [UZ 2170](https://phoible.org/inventories/view/2170) |
| [`s`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/s.mp3) | [afkoeling`s`periode](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/afkoelingsperiode.mp3) | ɑ f k u l ɪ ŋ `s` p e r i j o d ə |  |
| [`t`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/t.mp3) | [fou`t`enmarge](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/foutenmarge.mp3) | f ɯ `t` ə m ɑ r ʒ ə |  |
| [`u`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/u.mp3) | [afk`oe`lingsperiode](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/afkoelingsperiode.mp3) | ɑ f k `u` l ɪ ŋ s p e r i j o d ə |  |
| [`v`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/v.mp3) | [`v`rachtwagenchauffeurs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/vrachtwagenchauffeurs.mp3) | `v` r ɑ x t ʋ a ɣ ə n ʃ o f ø r s |  |
| [`x`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/x.mp3) | [vra`ch`twagenchauffeurs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/vrachtwagenchauffeurs.mp3) | v r ɑ `x` t ʋ a ɣ ə n ʃ o f ø r s |  |
| [`y`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/y.mp3) | [meerderheidsbeslu`i`tvorming](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/meerderheidsbesluitvorming.mp3) | m e r d ə r h ɛ i d z b ə s l œ `y` t f ɔ r m ɪ ŋ |  |
| [`z`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/z.mp3) | [meerderheid`s`besluitvorming](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/meerderheidsbesluitvorming.mp3) | m e r d ə r h ɛ i d `z` b ə s l œ y t f ɔ r m ɪ ŋ |  |
| [`ʒ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʒ.mp3) | [foutenmar`g`e](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/nl/word_audio/foutenmarge.mp3) | f ɯ t ə m ɑ r `ʒ` ə |  |

