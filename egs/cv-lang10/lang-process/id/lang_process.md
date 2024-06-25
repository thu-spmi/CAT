# Indonesian
Author: Ma, Te (mate153125@gmail.com)
## 1. Text normalization 

(1) Before genetaring lexicon, we need to normalize text. The code of text normalization for __Indonesian__ is in the script named [`text_norm.sh`](text_norm.sh).

## 2. Lexicon generation and correction

We use the FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit, [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), to create the pronunciation lexicon. The trained FSTs for use with Phonetisaurus is provided in [LanguageNet](https://github.com/uiuc-sst/g2ps#languagenet-grapheme-to-phoneme-transducers).

Note that the above G2P procedure is not perfect. As noted in `LanguageNet`, "PERs range from 7% to 45%".
The G2P-generated lexicon needs to be corrected. The correction step is based on [the LanguageNet symbol table for __Indonesian__](https://github.com/uiuc-sst/g2ps/blob/masterIndonesian/Indonesian_wikipedia_symboltable.html). The code of this step of lexicon correction is in the script named [`lexicon.sh`](lexicon.sh).

(1) We remove some special symbols such as accent symbols to enable sharing more phonemes between different languages.

| Removed symbols | Note |
| ------ | ------ |
| `ː` | Accent | 
| `ˈ` | Long vowel | 
| `ˌ` | Syllable |

## 3. Check of phonemes

Strictly speaking, one phoneme might correspond to multiple phones (those phones are referred to as the allophones). Note that our above procedure removes the diacritic, the notion of phonemes in this work is a looser one.

The generated lexicon from the G2P procedure is named [`lexicon_id.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ky/lexicon_id.txt). The set of IPA phonemes appeared in the lexicon is saved in [`phone_list.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/ky/phone_list.txt). We further check `phone_list.txt`, by referring to the following two phoneme lists and with listening tests.  

* IPA symbol table in LanguageNet, which, thought by LanguageNet, contains all the phones in the language:
https://github.com/uiuc-sst/g2ps/blob/masterIndonesian/Indonesian_wikipedia_symboltable.html
  
* IPA symbol table in Phoible: 
https://phoible.org/languages/indo1316. For each language, there may exist multiple phoneme inventories, which are archived at the Phoible website. 
For __Indonesian__, we choose the first one as the main reference for phoneme checking, which is [PH 1144](https://phoible.org/inventories/view/1144).

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

| IPA symbol in `phone_list.txt` | Word |  <div style="width: 150pt">G2P labeling result | Note |
| ------ | ------ | ------ | ------ |
| [`ɣ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɣ.mp3) | [fulbri`gh`t](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/fulbright.mp3) | f u l b r i `ɣ` t | Incorrect G2P labeling. The phoneme `/ɣ/` is not contained in any phoneme tables of Phoible, and the phoneme labeling should be corrected to `/f u l b r a i t/` |
| [`ʃ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʃ.mp3) | [ber`sy`ukur](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/bersyukur.mp3) | b e r `ʃ` u k o r |  |
| [`ɡ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɡ.mp3) | [elektroma`g`netisme](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/elektromagnetisme.mp3) | ə l e k t r o m a `ɡ` n ɪ t i s m ɛ |  |
| [`ʊ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʊ.mp3) | [abr`u`zzi](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/abruzzi.mp3) | ə b r `ʊ` z s i | Incorrect G2P labeling. The phoneme `/ʊ/` is not contained in any phoneme tables of Phoible, and the phoneme labeling should be corrected to `/a b r ʊ z z i/` |
| [`ɲ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɲ.mp3) | [payung`ny`a](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/payungnya.mp3) | p ə j ʊ ŋ `ɲ` ə |  |
| [`ɪ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɪ.mp3) | [elektromagnet`i`sme](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/elektromagnetisme.mp3) | ə l e k t r o m a ɡ n `ɪ` t i s m ɛ | The phoneme `/ɪ/` is not contained in any phoneme tables of Phoible, but sounds correct |
| [`a`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/a.mp3) | [elektrom`a`gnetisme](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/elektromagnetisme.mp3) | ə l e k t r o m `a` ɡ n ɪ t i s m ɛ |  |
| [`b`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/b.mp3) | [`b`everwijck](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/beverwijck.mp3) | `b` ə v ɪ r w i d͡ʒ t͡ʃ k | Incorrect G2P labeling. The phoneme labeling should be corrected to `/b e v ə r w i k/` |
| [`d`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/d.mp3) | [neighbourhoo`d`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/neighbourhood.mp3) | n ə e x b ʊ u r h o ɔ `d` |  |
| [`dʒ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/dʒ.mp3) | [beverwi`j`ck](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/beverwijck.mp3) | b ə v ɪ r w i `d͡ʒ` t͡ʃ k | Incorrect G2P labeling. The phoneme labeling should be corrected to `/b e v ə r w i k/` |
| [`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/e.mp3) | [el`e`ktromagnetisme](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/elektromagnetisme.mp3) | ə l `e` k t r o m a ɡ n ɪ t i s m ɛ |  |
| [`ə`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ə.mp3) | [`e`lektromagnetisme](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/elektromagnetisme.mp3) | `ə` l e k t r o m a ɡ n ɪ t i s m ɛ |  |
| [`ɛ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɛ.mp3) | [elektromagnetism`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/elektromagnetisme.mp3) | ə l e k t r o m a ɡ n ɪ t i s m `ɛ` | The phoneme `/ɛ/` is not contained in [PH 1144](https://phoible.org/inventories/view/1144), but contained in [GM 1609](https://phoible.org/inventories/view/1690) |
| [`f`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/f.mp3) | [`f`ulbright](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/fulbright.mp3) | `f` u l b r i ɣ t |  |
| [`h`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/h.mp3) | [neighbour`h`ood](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/neighbourhood.mp3) | n ə e x b ʊ u r `h` o ɔ d |  |
| [`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/i.mp3) | [abruzz`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/abruzzi.mp3) | ə b r ʊ z s `i` | Incorrect G2P labeling. The  phoneme labeling should be corrected to `/a b r ʊ z z i/` |
| [`j`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/j.mp3) | [pa`y`ungnya](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/payungnya.mp3) | p ə `j` ʊ ŋ ɲ ə |  |
| [`k`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/k.mp3) | [ele`k`tromagnetisme](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/elektromagnetisme.mp3) | ə l e `k` t r o m a ɡ n ɪ t i s m ɛ |  |
| [`l`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/l.mp3) | [e`l`ektromagnetisme](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/elektromagnetisme.mp3) | ə `l` e k t r o m a ɡ n ɪ t i s m ɛ |  |
| [`m`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/m.mp3) | [elektro`m`agnetis`m`e](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/elektromagnetisme.mp3) | ə l e k t r o `m` a ɡ n ɪ t i s `m` ɛ |  |
| [`n`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/n.mp3) | [`n`eighbourhood](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/neighbourhood.mp3) | `n` ə e x b ʊ u r h o ɔ d | Incorrect G2P labeling. The phoneme labeling should be corrected to `/n e i b ʊ u r h ʊ d/` |
| [`ŋ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ŋ.mp3) | [payu`ng`nya](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/payungnya.mp3) | p ə j ʊ `ŋ` ɲ ə |  |
| [`o`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/o.mp3) | [elektr`o`magnetisme](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/elektromagnetisme.mp3) | ə l e k t r `o` m a ɡ n ɪ t i s m ɛ |  |
| [`ɔ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɔ.mp3) | [neighbourho`o`d](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/neighbourhood.mp3) | n ə e x b ʊ u r h o `ɔ` d | Incorrect G2P labeling. The phoneme `/ɔ/` is not contained in any phoneme tables of Phoible, and the phoneme labeling should be corrected to `/n e i b ʊ u r h ʊ d/` |
| [`p`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/p.mp3) | [`p`ayungnya](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/payungnya.mp3) | `p` ə j ʊ ŋ ɲ ə |  |
| [`q`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/q.mp3) | [a`q`uilonians](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/aquilonians.mp3) | ə `q` u e l ʊ n e ə n s | Incorrect G2P labeling. The phoneme `/q/` is not contained in any phoneme tables of Phoible, and the phoneme labeling should be corrected to `/a k u i l ʊ n i a n s/` |
| [`r`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/r.mp3) | [elekt`r`omagnetisme](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/elektromagnetisme.mp3) | ə l e k t `r` o m a ɡ n ɪ t i s m ɛ |  |
| [`s`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/s.mp3) | [abruz`z`i](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/abruzzi.mp3) | ə b r ʊ z `s` i | Incorrect G2P labeling. The letter `z` is pronounced as `/z/`, and the phoneme labeling should be corrected to `/a b r ʊ z z i/` |
| [`t`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/t.mp3) | [fulbrigh`t`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/fulbright.mp3) | f u l b r i ɣ `t` |  |
| [`tʃ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/tʃ.mp3) | [beverwij`c`k](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/beverwijck.mp3) | b ə v ɪ r w i d͡ʒ `t͡ʃ` k |  |
| [`u`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/u.mp3) | [f`u`lbright](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/fulbright.mp3) | f `u` l b r i ɣ t |  |
| [`v`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/v.mp3) | [be`v`erwijck](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/beverwijck.mp3) | b ə `v` ɪ r w i d͡ʒ t͡ʃ k | The phoneme `/v/` is not contained in [PH 1144](https://phoible.org/inventories/view/1144), but contained in [GM 1609](https://phoible.org/inventories/view/1690) |
| [`w`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/w.mp3) | [bever`w`ijck](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/beverwijck.mp3) | b ə v ɪ r `w` i d͡ʒ t͡ʃ k |  |
| [`x`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/x.mp3) | [neig`h`bourhood](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/neighbourhood.mp3) | n ə e `x` b ʊ u r h o ɔ d | Incorrect G2P labeling. The phoneme labeling should be corrected to `/n e i b ʊ u r h ʊ d/` |
| [`z`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/z.mp3) | [abru`z`zi](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/id/word_audio/abruzzi.mp3) | ə b r ʊ `z` s i | Incorrect G2P labeling. The phoneme labeling should be corrected to `/a b r ʊ z z i/` |
