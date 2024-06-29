# English
Author: Ma, Te (mate153125@gmail.com)
## 1. Text normalization 

(1) The G2P models cannot recognize alien words, so we choose to remove sentences that containing alien words. They are listed in the file [`English_alien_sentences.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/data-process/en/English_alien_sentences.txt).

(2) Before creating lexicon, we need to normalize text. The code of text normalization for __English__ is in the script named [`text_norm.sh`](./text_norm.sh).

## 2. Lexicon generation and correction

We use the FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit, [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), to create the pronunciation lexicon. The trained FSTs for use with Phonetisaurus is provided in [LanguageNet](https://github.com/uiuc-sst/g2ps#languagenet-grapheme-to-phoneme-transducers).

Note that the above G2P procedure is not perfect. As noted in `LanguageNet`, "PERs range from 7% to 45%".
The G2P-generated lexicon needs to be corrected. The correction step is based on [the LanguageNet symbol table for __English__](https://github.com/uiuc-sst/g2ps/blob/master/English-US/English-US_wikipedia_symboltable.txt). The code of this step of lexicon correction is in the script named [`lexicon.sh`](./lexicon.sh).

(1) We remove some special symbols such as accent symbols to enable sharing more phonemes between different languages.

| Removed symbols | Note |
| ------ | ------ |
| `ː` | Accent | 
| `ˈ` | Long vowel |
| `ˌ` | Syllable |
| `#` | redundant |
| `.` | redundant |

(2) A subtle issue is that IPA symbols may be encoded in different forms. So to enforce consistency, the phoneme `/g/` is corrected to `/ɡ/`. And we remove the diacritic, with the motivation that coarse granularity may alleviate data scarcity issues and facilitate sharing between languages. For the phoneme `/t͡ʃ/`, we need to revise and `/tʃ/` to a single phoneme `/t͡ʃ/`. So as for `/d͡ʒ/`. And for other diphones with only one sound, we split them into two single phonemes.

| Phonemes from G2P | Phonemes corrected |
| ------ | ------ |
| `g` | `ɡ` |
| `n̩` | `n` |
| `l̩` | `l` |
| `ɝ` | `ɜ` |
| `ɚ` | `ə` |
| `tʃ` | `t͡ʃ` |
| `dʒ` | `d͡ʒ` |
| `d ʒ` | `d͡ʒ` |
| `ei` | `e i` |
| `aɪ` | `a ɪ` |
| `ɔi` | `ɔ i` |
| `oʊ` | `o ʊ` |
| `aʊ` | `a ʊ` |
| `ɑɪ` | `ɑ ɪ` |
| `ɔɪ` | `ɔ ɪ` |

(3) There are some special words such as "C++", which G2P models cannot recognize correctly, so we need to correct these phonemes. The symbol '-' in the following table means that the phoneme was missed.

| Word | Phonemes from G2P | Phonemes corrected |
| ------ | ------ | ------ |
| b | `b` | `b i` |
| c | `k` | `si` |
| c's | `k ə z` | `s i z` |
| c++ | `k` | `s i p l ʌ s p l ʌ s` |
| ch | `t͡ʃ` | `s i e i t͡ʃ` |
| d | `d` | `d i` |
| dz | `d z` | `d i z i` |
| e | - | `i` |
| g | - | `d͡ʒ i` |
| h | - | `e i t͡ʃ` |
| h'm | `h m` | `e i t͡ʃ ɛ m` |
| i | - | `ɑ ɪ` |
| j | - | `d͡ʒ e i` |
| k | - | `k e i` |
| l | - | `ə l` |
| n | - | `ə n` |
| o | - | `o ʊ` |
| q | - | `k j u` |
| q's | `k z` | `k j u z` |
| r | - | `ɑ ɹ` |
| s | - | `ɛ s` |
| t | - | `t i` |
| f | f | `ɛ f` |
| p | - | `p i` |
| p's | `p z` | `p i z` |
| pz | `p z` | `p i z i` |
| qa | `k ə` | `k j u e i` |
| qb | `k b` | `k j u b i` |
| qbert | `k b ə t` | `k j u b ə t` |
| qi | `k i` | `t͡ʃ i` |
| xi | `z ɑ ɪ` | `ʃ i` |
| s's | `s z` | `ɛ s z` |
| m | `m` | `ɛ m` |
| u | - | `j u` |
| u's | `j z` | `j u z` |
| v | `v` | `v i` |
| v's | `v z` | `v i z`|
| vz | `v z` | `v i z i` |
| w | - | `d ʌ b l j u` |
| w's | `w z` | `d ʌ b l j u z` |
| x | - | `ɛ k s` |
| x's | `z z` | `ɛ k s z` |
| y | - | `w ɑ ɪ` |
| y's | `j z` | `w ɑ ɪ z` |

## 3. Check of phonemes

Strictly speaking, one phoneme might correspond to multiple phones (those phones are referred to as the allophones). Note that our above procedure removes the diacritic, the notion of phonemes in this work is a looser one.

The generated lexicon from the G2P procedure is named [`lexicon_en.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/lexicon_en.txt). The set of IPA phonemes appeared in the lexicon is saved in [`phone_list.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/phone_list.txt). We further check `phone_list.txt`, by referring to the following two phoneme lists and with listening tests.  

* IPA symbol table in LanguageNet, which, thought by LanguageNet, contains all the phones in the language:
https://github.com/uiuc-sst/g2ps/blob/master/English-US/English-US_wikipedia_symboltable.txt
  
* IPA symbol table in Phoible: 
https://phoible.org/languages/stan1293. For each language, there may exist multiple phoneme inventories, which are archived at the Phoible website. 
For __English__, we choose the first one as the main reference for phoneme checking, which is [SPA 160](https://phoible.org/inventories/view/160).

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

| IPA symbol in `phone_list.txt` | Word | <div style="width: 120pt">G2P labeling result | Note |
| ------ | ------ | ------ | ------ |
| [`ɑ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɑ.mp3) | [`aa`rthakshathe](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/aarthakshathe.mp3) | `ɑ` ɹ ɵ ə k ʃ e i ð | The phoneme `/ɑ/` is not contained in [SPA 160](https://phoible.org/inventories/view/160), but contained in [UZ 2175](https://phoible.org/inventories/view/2175) |
| [`ʊ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʊ.mp3) | [lapph`u`nds](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/lapphunds.mp3) | l æ p h `ʊ` n t s | The letter `h` is silent, so the correct phoneme labeling should be `/l æ p ʊ n t s/` |
| [`ɹ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɹ.mp3) | [aa`r`thakshathe](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/aarthakshathe.mp3) | ɑ `ɹ` ɵ ə k ʃ e i ð | The phoneme `/ɹ/` is not contained in [SPA 160](https://phoible.org/inventories/view/160), but contained in [UZ 2175](https://phoible.org/inventories/view/2175) |
| [`ɡ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɡ.mp3) | [abne`g`ation](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abnegation.mp3) | æ b n ɪ `ɡ` e i ʃ n | |
| [`ɾ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɾ.mp3) | [aphani`t`ic](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/aphanitic.mp3) | æ f ə n ɪ `ɾ` ɪ k | Incorrect G2P labeling. The letter `t` is pronounced as `/t/`, and the phoneme `/ɾ/` is not contained in any phoneme tables of LanguageNet or Phoible, so it needs to be corrected to `/t/` |
| [`ɪ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɪ.mp3) | [aphanit`i`c](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/aphanitic.mp3) | æ f ə n ɪ ɾ `ɪ` k |  |
| [`ʌ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʌ.mp3) | [abd`u`ction](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abduction.mp3) | æ b d `ʌ` k ʃ n | The phoneme `/ʌ/` is not contained in [SPA 160](https://phoible.org/inventories/view/160), but contained in [UZ 2175](https://phoible.org/inventories/view/2175) |
| [`ɜ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɜ.mp3) | [p`e`rnambuco](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/pernambuco.mp3) | p `ɜ` ɹ n ə m b j u k o ʊ | The phoneme `/ɜ/` is not contained in [SPA 160](https://phoible.org/inventories/view/160), but contained in [UZ 2177](https://phoible.org/inventories/view/2177) |
| [`ʃ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʃ.mp3) | [aarthak`sh`athe](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/aarthakshathe.mp3) | ɑ ɹ ɵ ə k `ʃ` e i ð |  |
| [`a`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/a.mp3) | [meadowm`o`unt](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/meadowmount.mp3) | m ɛ d o ʊ m `a` ʊ n t |  |
| [`æ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/æ.mp3) | [`a`bnegation](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abnegation.mp3) | `æ` b n ɪ ɡ e i ʃ n | |
| [`b`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/b.mp3) | [a`b`negation](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abnegation.mp3) | æ `b` n ɪ ɡ e i ʃ n | |
| [`d`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/d.mp3) | [ab`d`uction](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abduction.mp3) | æ b `d` ʌ k ʃ n |  |
| [`ð`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ð.mp3) | [aarthaksha`th`e](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/aarthakshathe.mp3) | ɑ ɹ ɵ ə k ʃ e i `ð` |  |
| [`dʒ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/dʒ.mp3) | [ab`j`ad](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abjad.mp3) | æ b `dʒ` æ d |  |
| [`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/e.mp3) | [att`a`ining](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/attaining.mp3) | ə t `e` i n ɪ ŋ |  |
| [`ɛ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɛ.mp3) | [ab`e`llio](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abellio.mp3) | ə b `ɛ` l i o ʊ | Incorrect G2P labeling. The letter `e` is pronounced as `/i/`, and phoneme `/ɛ/` is not contained in [SPA 160](https://phoible.org/inventories/view/160), but contained in [UZ 2175](https://phoible.org/inventories/view/2175). For this word, it needs to be corrected to `/i/` after listening  |
| [`ə`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ə.mp3) | [`a`bellio](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abellio.mp3) | `ə` b ɛ l i o ʊ |  |
| [`f`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/f.mp3) | [aardwol`f`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/aardwolf.mp3) | ɑ ɹ d w ʊ l `f` |  |
| [`h`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/h.mp3) | [aba`j`o](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abajo.mp3) | ə b ɑ `h` o ʊ |  |
| [`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/i.mp3) | [abell`i`o](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abellio.mp3) | ə b ɛ l `i` o ʊ |  |
| [`j`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/j.mp3) | [pernamb`u`co](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/pernambuco.mp3) | p ɜ ɹ n ə m b `j` u k o ʊ | Incorrect G2P labeling. The redundant phoneme `/j/` should be removed |
| [`k`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/k.mp3) | [abdu`c`tion](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abduction.mp3) | æ b d ʌ `k` ʃ n |  |
| [`l`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/l.mp3) | [abe`ll`io](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abellio.mp3) | ə b ɛ `l` i o ʊ |  |
| [`m`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/m.mp3) | [meadow`m`ount](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/meadowmount.mp3) | m ɛ d o ʊ `m` a ʊ n t |  |
| [`n`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/n.mp3) | [apha`n`itic](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/aphanitic.mp3) | æ f ə `n` ɪ ɾ ɪ k |  |
| [`ŋ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ŋ.mp3) | [attaini`ng`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/attaining.mp3) | ə t e i n ɪ `ŋ` |  |
| [`o`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/o.mp3) | [mead`o`wmount](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/meadowmount.mp3) | m ɛ d `o` ʊ m a ʊ n t |  |
| [`ɵ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɵ.mp3) | [aar`th`akshathe](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/aarthakshathe.mp3) | ɑ ɹ `ɵ` ə k ʃ e i ð | The phoneme `/ɛ/` is not contained in [SPA 160](https://phoible.org/inventories/view/160), but contained in [UZ 2179](https://phoible.org/inventories/view/2179) |
| [`ɔ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɔ.mp3) | [`au`diovisual](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/audiovisual.mp3) | `ɔ` d i o ʊ v ɪ ʒ u l |  |
| [`p`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/p.mp3) | [la`pp`hunds](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/lapphunds.mp3) | l æ `p` h ʊ n t s | The letter `h` is silent, so the correct phoneme labeling should be `/l æ p ʊ n t s/` |
| [`s`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/s.mp3) | [ac`c`enture](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/accenture.mp3) | æ k `s` ɛ n tʃ ə |  |
| [`t`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/m.mp3) | [meadowmoun`t`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/meadowmount.mp3) | m ɛ d o ʊ m a ʊ n `t` |  |
| [`tʃ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/tʃ.mp3) | [accen`tu`re](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/accenture.mp3) | æ k s ɛ n `t͡ʃ` ə | The same pronunciation for `/tʃ/` and `/t͡ʃ/`, so replacing `/tʃ/` with `/t͡ʃ/` |
| [`u`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/u.mp3) | [pernamb`u`co](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/pernambuco.mp3) | p ɜ ɹ n ə m b j `u` k o ʊ |  |
| [`v`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/v.mp3) | [audio`v`isual](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/audiovisual.mp3) | ɔ d i o ʊ `v` ɪ ʒ u l |  |
| [`w`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/w.mp3) | [aard`w`olf](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/aardwolf.mp3) | ɑ ɹ d `w` ʊ l f |  |
| [`z`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/z.mp3) | [aba`z`a](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/abaza.mp3) | ə b ɑ `z` ə |  |
| [`ʒ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʒ.mp3) | [audiovi`s`ual](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/en/word_audio/audiovisual.mp3) | ɔ d i o ʊ v ɪ `ʒ` u l |  |



