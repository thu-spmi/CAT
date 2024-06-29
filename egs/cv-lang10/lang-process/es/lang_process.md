# Spanish
Author: Ma, Te (mate153125@gmail.com)
## 1. Text normalization 

(1) The G2P models cannot recognize alien words, so we choose to remove sentences that containing alien words. They are listed in the file [`Spanish_alien_sentences.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/data-process/es/Spanish_alien_sentences.txt).

(2) Before creating lexicon, we need to normalize text. The code of text normalization for __Spanish__ is in the script named [`text_norm.sh`](./text_norm.sh).

## 2. Lexicon generation and correction

We use the FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit, [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), to create the pronunciation lexicon. The trained FSTs for use with Phonetisaurus is provided in [LanguageNet](https://github.com/uiuc-sst/g2ps#languagenet-grapheme-to-phoneme-transducers).

Note that the above G2P procedure is not perfect. As noted in `LanguageNet`, "PERs range from 7% to 45%".
The G2P-generated lexicon needs to be corrected. The correction step is based on [the LanguageNet symbol table for __Spanish__](https://github.com/uiuc-sst/g2ps/blob/master/Spanish/Spanish_wikipedia_symboltable.txt). The code of this step of lexicon correction is in the script named [`lexicon.sh`](./lexicon.sh).

(1) We remove some special symbols such as accent symbols to enable sharing more phonemes between different languages. 

| Removed symbols | Note |
| ------ | ------ |
| `ː` | Accent | 
| `ˈ` | Long vowel |
| `ʲ` | Velarization |

(2) A subtle issue is that IPA symbols may be encoded in different forms. So to enforce consistency, the phoneme `/g/` is corrected to `/ɡ/`.

| Phonemes from G2P | Phonemes corrected |
| ------ | ------ |
| `g` | `ɡ` |



## 3. Check of phonemes

Strictly speaking, one phoneme might correspond to multiple phones (those phones are referred to as the allophones). Note that our above procedure removes the diacritic, the notion of phonemes in this work is a looser one.

The generated lexicon from the G2P procedure is named [`lexicon_es.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/lexicon_es.txt). The set of IPA phonemes appeared in the lexicon is saved in [`phone_list.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/phone_list.txt). We further check `phone_list.txt`, by referring to the following two phoneme lists and with listening tests.  

* IPA symbol table in LanguageNet, which, thought by LanguageNet, contains all the phones in the language:
https://github.com/uiuc-sst/g2ps/blob/master/Spanish/Spanish_wikipedia_symboltable.txt
  
* IPA symbol table in Phoible: 
https://phoible.org/languages/stan1288. For each language, there may exist multiple phoneme inventories, which are archived at the Phoible website. 
For __Spanish__, we choose the first one as the main reference for phoneme checking, which is [SPA 164](https://phoible.org/inventories/view/164).

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

| IPA symbol in `phone_list.txt` | Word | <div style="width: 150pt">G2P labeling result | Note |
| ------ | ------ | ------ | ------ |
| [`ɱ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɱ.mp3) | [abraha`m`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/abraham.mp3) | a β ɾ a `ɱ` | Incorrect G2P labeling. The phoneme `/ɱ/` is not contained in any phoneme tables of LanguageNet or Phoible, and the phoneme labeling should be corrected to `/a β ɾ a a m/` after listening.  |
| [`ɲ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɲ.mp3) | [argando`ñ`a](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/argandon%CC%83a.mp3) | a ɾ ɡ a n d o `ɲ` a |  |
| [`ɣ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɣ.mp3) | [acha`g`uas](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/achaguas.mp3) | a c a `ɣ` w u a s | Incorrect G2P labeling. The phoneme `/ɣ/` is not contained in phoneme tables of LanguageNet, and needs to be corrected to `/g/` after listening |
| [`ɡ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɡ.mp3) | [atxa`g`a](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/atxaga.mp3) | a t͡ʃ a `ɡ` a | The phoneme `/ɡ/` is not contained in [SPA 164](https://phoible.org/inventories/view/164), but contained in [UZ 2210](https://phoible.org/inventories/view/2210) |
| [`ɾ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɾ.mp3) | [ab`r`aham](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/abraham.mp3) | a β `ɾ` a ɱ |  |
| [`ʎ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʎ.mp3) | [fe`ll`owship](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/fellowship.mp3) | f e `ʎ` o w ʃ j p | The phoneme `/ʎ/` is not contained in phoneme tables of LanguageNet, but it sounds correct |
| [`ʃ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʃ.mp3) | [fellow`sh`ip](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/fellowship.mp3) | f e ʎ o w `ʃ` j p | The phoneme `/ʃ/` is not contained in [SPA 164](https://phoible.org/inventories/view/164), but contained in [EA 2308](https://phoible.org/inventories/view/2308) |
| [`a`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/a.mp3) | [`a`braham](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/abraham.mp3) | `a` β ɾ a ɱ |  |
| [`b`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/b.mp3) | [`b`endecir](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/bendecir.mp3) | `b` e n d e θ i ɾ | The phoneme `/b/` is not contained in [SPA 164](https://phoible.org/inventories/view/164), but contained in [UZ 2210](https://phoible.org/inventories/view/2210) |
| [`c`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/c.mp3) | [a`ch`aguas](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/achaguas.mp3) | a `c` a ɣ w u a s | Incorrect G2P labeling. The phoneme `/c/` is not contained in any phoneme tables of LanguageNet or Phoible, and needs to be corrected to `/t͡ʃ/` after listening |
| [`d`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/d.mp3) | [argan`d`oña](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/argandoña.mp3) | a ɾ ɡ a n `d` o ɲ a | The phoneme `/d/` is not contained in [SPA 164](https://phoible.org/inventories/view/164), but contained in [UZ 2210](https://phoible.org/inventories/view/2210) |
| [`ð`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ð.mp3) | [interjuris`d`iccional](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/interjurisdiccional.mp3) | i n t e ɾ x u ɾ i z `ð` i k s i o n a l | Incorrect G2P labeling. The phoneme `/ð/` is not contained in phoneme tables of LanguageNet, and needs to be corrected to `/d/` after listening |
| [`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/e.mp3) | [f`e`llowship](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/fellowship.mp3) | f `e` ʎ o w ʃ j p |  |
| [`f`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/f.mp3) | [`f`ellowship](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/fellowship.mp3) | `f` e ʎ o w ʃ j p |  |
| [`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/i.mp3) | [bendec`i`r](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/bendecir.mp3) | b e n d e θ `i` ɾ |  |
| [`j`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/j.mp3) | [fellowsh`i`p](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/fellowship.mp3) | f e ʎ o w ʃ `j` p | Incorrect G2P labeling. The phoneme `/j/` is not contained in phoneme tables of LanguageNet, and needs to be corrected to `/i/` after listening |
| [`k`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/k.mp3) | [barran`c`abermeja](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/barrancabermeja.mp3) | b a r a n `k` a β e ɾ m e x a |  |
| [`l`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/l.mp3) | [interjurisdicciona`l`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/interjurisdiccional.mp3) | i n t e ɾ x u ɾ i z ð i k s i o n a `l` |  |
| [`m`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/m.mp3) | [barrancaber`m`eja](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/barrancabermeja.mp3) | b a r a n k a β e ɾ `m` e x a |  |
| [`n`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/n.mp3) | [interjurisdiccio`n`al](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/interjurisdiccional.mp3) | i n t e ɾ x u ɾ i z ð i k s i o `n` a l |  |
| [`o`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/o.mp3) | [fell`o`wship](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/fellowship.mp3) | f e ʎ `o` w ʃ j p |  |
| [`p`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/p.mp3) | [fellowshi`p`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/fellowship.mp3) | f e ʎ o w ʃ j `p` |  |
| [`r`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/r.mp3) | [bar`r`ancabermeja](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/barrancabermeja.mp3) | b a `r` a n k a β e ɾ m e x a |  |
| [`s`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/s.mp3) | [achagua`s`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/achaguas.mp3) | a c a ɣ w u a `s` |  |
| [`t`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/t.mp3) | [in`t`erjurisdiccional](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/interjurisdiccional.mp3) | i n `t` e ɾ x u ɾ i z ð i k s i o n a l |  |
| [`tʃ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/tʃ.mp3) | [a`tx`aga](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/atxaga.mp3) | a `t͡ʃ` a ɡ a | The same pronunciation for `/tʃ/` and `/t͡ʃ/`, so replacing `/tʃ/` with `/t͡ʃ/` |
| [`u`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/u.mp3) | [achag`u`as](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/achaguas.mp3) | a c a ɣ w `u` a s | Incorrect G2P labeling. The letter `gua` is pronounced as `/g w a/`, so the phoneme labeling should be corrected to `/a t͡ʃ a g w a s/`  |
| [`w`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/w.mp3) | [achag`u`as](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/achaguas.mp3) | a c a ɣ `w` u a s |  |
| [`x`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/x.mp3) | [barrancaberme`j`a](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/barrancabermeja.mp3) | b a r a n k a β e ɾ m e `x` a |  |
| [`z`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/z.mp3) | [interjuri`s`diccional](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/interjurisdiccional.mp3) | i n t e ɾ x u ɾ i `z` ð i k s i o n a l | Incorrect G2P labeling. The phoneme `/z/` is not contained in any phoneme tables of LanguageNet or Phoible, and the letter `s` is pronounced as `/s/` so the phoneme `/z/` needs to be corrected to `/s/` |
| [`β`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/β.mp3) | [a`b`raham](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/abraham.mp3) | a `β` ɾ a ɱ | The phoneme `/β/` is not contained in phoneme tables of LanguageNet, but it sounds correct |
| [`θ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/θ.mp3) | [bende`c`ir](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/es/word_audio/bendecir.mp3) | b e n d e `θ` i ɾ | Incorrect G2P labeling. The phoneme `/θ/` is not contained in phoneme tables of LanguageNet, and the letter `c` is pronounced as `/s/` so the phoneme `/θ/` needs to be corrected to `/s/` |
