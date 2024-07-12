# French
Author: Ma, Te (mate153125@gmail.com)
## 1. Text normalization 

(1) The G2P models cannot recognize alien words, so we choose to remove sentences that containing alien words. They are listed in the file [`French_alien_sentences.txt`](/home/mate/cat_multilingual/egs/cv-lang10/data-process/fr/French_alien_sentences.txt).

(2) Before creating lexicon, we need to normalize text. The code of text normalization for __French__ is in the script named [`text_norm.sh`](./text_norm.sh).

## 2. Lexicon generation and correction

We use the FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit, [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), to create the pronunciation lexicon. The trained FSTs for use with Phonetisaurus is provided in [LanguageNet](https://github.com/uiuc-sst/g2ps#languagenet-grapheme-to-phoneme-transducers).

Note that the above G2P procedure is not perfect. As noted in `LanguageNet`, "PERs range from 7% to 45%".
The G2P-generated lexicon needs to be corrected. The correction step is based on [the LanguageNet symbol table for __French__](https://github.com/uiuc-sst/g2ps/blob/master/French/French_wikipedia_symboltable.txt). The code of this step of lexicon correction is in the script named [`lexicon.sh`](./lexicon.sh).

(1) We remove some special symbols such as accent symbols to enable sharing more phonemes between different languages.

| Removed symbols | Note |
| ------ | ------ |
| `ː` | Accent | 
| `ˈ` | Long vowel |
| `ˌ` | Syllable |

(2) For some letters that cannot be recognized by the G2P model, the unrecognized letters will be directly output as they are in the G2P labeling result, as shown below. These error labels are corrected according to the pronunciation of these letters.

| Phoneme from G2P | Phoneme corrected |
| ------ | ------ |
| `g` | `ʒ` |
| `R` | `ʁ` |
| `í` | `i` |
| `ì` | `i` |
| `ò` | `o` |
| `ó` | `o` |
| `ü` | `u` |
| `ú` | `u` |
| `ù` | `u` |
| `á` | `a` |
| `ɑ̃` | `ɑ` |
| `œ̃` | `œ` |
| `ɛ̃` | `ɛ` |
| `ÿ` | `y` |
| `ë` | `e` |
| `ɔ̃` | `ɔ` |

## 3. Check of phonemes

Strictly speaking, one phoneme might correspond to multiple phones (those phones are referred to as the allophones). Note that our above procedure removes the diacritic, the notion of phonemes in this work is a looser one.

The generated lexicon from the G2P procedure is named [`lexicon_fr.txt`](/home/mate/cat_multilingual/egs/cv-lang10/data-process/fr/lexicon_fr.txt). The set of IPA phonemes appeared in the lexicon is saved in [`phone_list.txt`](/home/mate/cat_multilingual/egs/cv-lang10/data-process/fr/phone_list.txt). We further check `phone_list.txt`, by referring to the following two phoneme lists and with listening tests.  

* IPA symbol table in LanguageNet, which, thought by LanguageNet, contains all the phones in the language:
https://github.com/uiuc-sst/g2ps/blob/master/French/French_wikipedia_symboltable.txt
  
* IPA symbol table in Phoible: 
https://phoible.org/languages/stan1290. For each language, there may exist multiple phoneme inventories, which are archived at the Phoible website. 
For __French__, we choose the first one as the main reference for phoneme checking, which is [SPA 162](https://phoible.org/inventories/view/162).

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

| IPA symbol in `phone_list.txt` | Word |  <div style="width: 140pt">G2P labeling result | Note |
| ------ | ------ | ------ | ------ |
| [`ʁ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʁ.mp3) | [g`r`avitationnellement](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/gravitationnellement.mp3) | ʒ `ʁ` a v i t a s j o n ɛ l ə m ɑ | The phoneme `/ʁ/` is not contained in [SPA 162](https://phoible.org/inventories/view/162), but contained in [UZ 2182](https://phoible.org/inventories/view/2182) |
| [`ɥ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɥ.mp3) | [affect`u`euse](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/affectueuse.mp3) | a f ɛ k t `ɥ` ø z |  |
| [`ɲ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɲ.mp3) | [acheilo`gn`athus](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/acheilognathus.mp3) | a ʃ ɛ l ɔ `ɲ` a t y |  |
| [`ɑ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɑ.mp3) | [gravitationnellement](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/gravitationnellement.mp3) | ʒ ʁ a v i t a s j o n ɛ l ə m `ɑ` | Incorrect G2P labeling. The letter `e` is pronounced as `/ə/`, so the phoneme `/ɑ/` needs to be corrected to `/ə/`, and the correct phoneme labeling should be `/ɡ ʁ a v i t a s i o n ɛ l ə m ə n/` after listening |
| [`ʃ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʃ.mp3) | [a`ch`eilognathus](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/acheilognathus.mp3) | a `ʃ` ɛ l ɔ ɲ a t y |  |
| [`a`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/a.mp3) | [`a`ffectueuse](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/affectueuse.mp3) | `a` f ɛ k t ɥ ø z |  |
| [`b`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/b.mp3) | [`b`atiengchaleunsouk](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/batiengchaleunsouk.mp3) | `b` a t j ɛ ŋ ʃ a l œ n s u k |  |
| [`d`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/d.mp3) | [poséi`d`ocroiseurs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/poséidocroiseurs.mp3) | p ɔ z e i `d` ɔ k ʁ w a z œ ʁ |  |
| [`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/e.mp3) | [pos`é`idocroiseurs](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/pose%CC%81idocroiseurs.mp3) | p ɔ z `e` i d ɔ k ʁ w a z œ ʁ |  |
| [`ɛ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɛ.mp3) | [aff`e`ctueuse](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/affectueuse.mp3) | a f `ɛ` k t ɥ ø z |  |
| [`ə`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ə.mp3) | [gravitationnell`e`ment](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/gravitationnellement.mp3) | ʒ ʁ a v i t a s j o n ɛ l `ə` m ɑ | The phoneme `/ə/` is not contained in [SPA 162](https://phoible.org/inventories/view/162), but contained in [UZ 2182](https://phoible.org/inventories/view/2182) |
| [`f`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/f.mp3) | [a`ff`ectueuse](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/affectueuse.mp3) | a `f` ɛ k t ɥ ø z |  |
| [`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/i.mp3) | [grav`i`tationnellement](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/gravitationnellement.mp3) | ʒ ʁ a v `i` t a s j o n ɛ l ə m ɑ |  |
| [`j`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/j.mp3) | [bat`i`engchaleunsouk](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/batiengchaleunsouk.mp3) | b a t `j` ɛ ŋ ʃ a l œ n s u k |  |
| [`k`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/k.mp3) | [affe`c`tueuse](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/affectueuse.mp3) | a f ɛ `k` t ɥ ø z |  |
| [`l`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/l.mp3) | [gravitationnel`l`ement](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/gravitationnellement.mp3) | ʒ ʁ a v i t a s j o n ɛ `l` ə m ɑ |  |
| [`m`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/m.mp3) | [gravitationnelle`m`ent](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/gravitationnellement.mp3) | ʒ ʁ a v i t a s j o n ɛ l ə `m` ɑ |  |
| [`n`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/n.mp3) | [gravitation`n`ellement](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/gravitationnellement.mp3) | ʒ ʁ a v i t a s j o `n` ɛ l ə m ɑ |  |
| [`ŋ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ŋ.mp3) | [batie`ng`chaleunsouk](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/batiengchaleunsouk.mp3) | b a t j ɛ `ŋ` ʃ a l œ n s u k |  |
| [`o`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/o.mp3) | [gravitati`o`nnellement](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/gravitationnellement.mp3) | ʒ ʁ a v i t a s j `o` n ɛ l ə m ɑ |  |
| [`ø`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ø.mp3) | [affectu`eu`se](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/affectueuse.mp3) | a f ɛ k t ɥ `ø` z |  |
| [`ɔ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɔ.mp3) | [acheil`o`gnathus](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/acheilognathus.mp3) | a ʃ ɛ l `ɔ` ɲ a t y |  |
| [`œ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/œ.mp3) | [poséidocrois`eu`rs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/poséidocroiseurs.mp3) | p ɔ z e i d ɔ k ʁ w a z `œ` ʁ |  |
| [`p`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/p.mp3) | [`p`oséidocroiseurs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/poséidocroiseurs.mp3) | `p` ɔ z e i d ɔ k ʁ w a z œ ʁ |  |
| [`s`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/s.mp3) | [batiengchaleun`s`ouk](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/batiengchaleunsouk.mp3) | b a t j ɛ ŋ ʃ a l œ n `s` u k |  |
| [`t`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/t.mp3) | [affec`t`ueuse](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/affectueuse.mp3) | a f ɛ k `t` ɥ ø z |  |
| [`u`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/u.mp3) | [batiengchaleuns`ou`k](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/batiengchaleunsouk.mp3) | b a t j ɛ ŋ ʃ a l œ n s `u` k |  |
| [`v`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/v.mp3) | [gra`v`itationnellement](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/gravitationnellement.mp3) | ʒ ʁ a `v` i t a s j o n ɛ l ə m ɑ |  |
| [`w`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/w.mp3) | [poséidocr`o`iseurs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/poséidocroiseurs.mp3) | p ɔ z e i d ɔ k ʁ `w` a z œ ʁ | The letter `oi` is pronounced as `/w a/` |
| [`x`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/x.mp3) | [`j`uanele](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/juanele.mp3) | `x` w a n ɛ l | Incorrect G2P labeling. The phoneme `/x/` is not contained in any phoneme tables of LanguageNet or Phoible, and the correct phoneme labeling should be `/w a n ɛ l/` after listening |
| [`y`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/y.mp3) | [acheilognath`u`s](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/acheilognathus.mp3) | a ʃ ɛ l ɔ ɲ a t `y` | Incorrect G2P labeling. The correct phoneme labeling should be `/a ʃ ɛ l ɔ ɲ a tʃ u s/` after listening |
| [`z`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/z.mp3) | [po`s`éidocroiseurs](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/poséidocroiseurs.mp3) | p ɔ `z` e i d ɔ k ʁ w a z œ ʁ |  |
| [`ʒ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʒ.mp3) | [`g`ravitationnellement](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/fr/word_audio/gravitationnellement.mp3) | `ʒ` ʁ a v i t a s j o n ɛ l ə m ɑ | Incorrect G2P labeling. The letter `g` is pronounced as `/ɡ/`, so the phoneme `/ʒ/` needs to be corrected to `/ɡ/`, and the correct phoneme labeling should be `/ɡ ʁ a v i t a s i o n ɛ l ə m ə n/` |



