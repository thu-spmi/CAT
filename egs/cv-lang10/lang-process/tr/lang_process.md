# Turkish
Author: Ma, Te (mate153125@gmail.com)
## 1. Text normalization 

Before creating lexicon, we need to normalize text. The code of text normalization for __Turkish__ is in the script named [`text_norm.sh`](./text_norm.sh).

## 2. Lexicon generation and correction

We use the FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit, [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), to create the pronunciation lexicon. The trained FSTs for use with Phonetisaurus is provided in [LanguageNet](https://github.com/uiuc-sst/g2ps#languagenet-grapheme-to-phoneme-transducers).

Note that the above G2P procedure is not perfect. As noted in `LanguageNet`, "PERs range from 7% to 45%".
The G2P-generated lexicon needs to be corrected. The correction step is based on [the LanguageNet symbol table for __Turkish__](https://github.com/uiuc-sst/g2ps/blob/masterTurkish/Turkish_wikipedia_symboltable.html). The code of this step of lexicon correction is in the script named [`lexicon.sh`](./lexicon.sh).

(1) We remove some special symbols such as accent symbols to enable sharing more phonemes between different languages.

| Removed symbols | Note |
| ------ | ------ |
| `ː` | Accent |
| `ˈ` | Long vowel |
| `ʲ` | Velarization |

(2) For the phoneme `/t͡ʃ/`, we need to revise `/t ʃ/` and `/tʃ/` to a single phoneme `/t͡ʃ/`. So as for `/d͡ʒ/`. The correction of `/ɡj/` is `/ɡʲ/`, which denotes a palatalized phoneme. We remove the diacritic, with the motivation that coarse granularity may alleviate data scarcity issues and facilitate sharing between languages.
A further subtle issue is that IPA symbols may be encoded in different forms. So to enforce consistency, the phoneme `/g/` is corrected to `/ɡ/`.

| Phoneme from G2P | Phoneme corrected |
| ------ | ------ |
| `t ʃ` | `t͡ʃ` |
| `tʃ` | `t͡ʃ` |
| `d ʒ` | `d͡ʒ` |
| `dʒ` | `d͡ʒ` |
| `ɡj` | `ɡ` |
| `g` | `ɡ` |

Note: We cannot change all the `/d ʒ/` to `/d͡ʒ/`, because there are some combination of letters that are pronounced as `/d ʒ/`, which are correct. For example,

| Word | Phoneme from G2P |
| ------ | ------ |
| tudjman'ın | t u d ʒ m a n ɨ n |

(3) For some letters that cannot be recognized by the G2P model, the unrecognized letters will be directly output as they are in the G2P labeling result, as shown below. These error labels are corrected according to the pronunciation of these letters.

| Unrecognized letter | Word | G2P labeling result | Correction |
| ------ | ------ | ------ | ------ |
| â | [harek`â`t](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/hareka%CC%82t.mp3) | h a ɾ e k `â` t | h a ɾ e k [`a`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/a.mp3) t |
| é | [charit`é`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/charite%CC%81.mp3) | d ʒ h a ɾ i t `é` | d ʒ h a ɾ i t [`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/e.mp3) |
| û | [mahk`û`m](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/mahku%CC%82m.mp3) | m a h k `û` m | m a h k [`u`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/u.mp3) m |
| ë | [vet`ë`vendosje](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/vete%CC%88vendosje.mp3) | v e t `ë` v e n d o s ʒ e | v e t [`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/e.mp3) v e n d o s ʒ e |
| î | [mûsik`î`ye](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/mu%CC%82siki%CC%82ye.mp3) | m û s i k `î` j e | m u s i k [`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/i.mp3) j e |

## 3. Check of phonemes

Strictly speaking, one phoneme might correspond to multiple phones (those phones are referred to as the allophones). Note that our above procedure removes the diacritic, the notion of phonemes in this work is a looser one.

The generated lexicon from the G2P procedure is named [`lexicon_tr.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/lexicon_tr.txt). The set of IPA phonemes appeared in the lexicon is saved in [`phone_list.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/phone_list.txt). We further check `phone_list.txt`, by referring to the following two phoneme lists and with listening tests.  

* IPA symbol table in LanguageNet, which, thought by LanguageNet, contains all the phones in the language:
https://github.com/uiuc-sst/g2ps/blob/masterTurkish/Turkish_wikipedia_symboltable.html
  
* IPA symbol table in Phoible: 
https://phoible.org/languages/nucl1301. For each language, there may exist multiple phoneme inventories, which are archived at the Phoible website. 
For __Turkish__, we choose the first one as the main reference for phoneme checking, which is [SPA 186](https://phoible.org/inventories/view/186).

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


| IPA symbol in `phone_list.txt` | Word | <div style="width: 160pt">G2P labeling result | Note |
| ------ | ------ | ------ | ------ |
| [`ʃ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʃ.ogg) | [dijitalle`ş`tirilmi`ş`tir](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/dijitalleştirilmiştir.mp3) | d i ʒ i t a ɫ l e `ʃ` t i r i l m i `ʃ` t i ɽ |  |
| [`ɡ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɡ.ogg) | [`g`örülmüyordu](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/görülmüyordu.mp3) | `ɡ` ø r y l m y j o r d u |  |
| [`ɾ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɾ.ogg) | [adalele`r`i](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/adaleleri.mp3) | a d a l e l e `ɾ` i |  |
| [`ɨ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɨ.ogg) | [abart`ı`yorsun](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/abartıyorsun.mp3) | a b a r t `ɨ` j o n | Incorrect G2P labeling. The phoneme `/ɨ/` is not contained in any phoneme tables of LanguageNet or Phoible, and needs to be corrected to `/ɯ/` after listening |
| [`ʔ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʔ.ogg) | [kur'an'da](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/kur'an'da.mp3) | k u r `ʔ` a n d a | Incorrect G2P labeling. The phoneme `/ʔ/` is redundant, so it needs to removed |
| [`ɟ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɟ.ogg) | [ba`g`iç'e](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/bagiç'e.mp3) | b a `ɟ` i t͡ʃ e | The letter `g` is pronounced as `/ɟ/` before letter `i`, `e`, `ö` or `ü`, otherwise it is pronounced as `/ɡ/` |
| [`ɽ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɽ.ogg) | [akmıyo`r`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/akmıyor.mp3) | a k m ɨ j o `ɽ` | Incorrect G2P labeling. The phoneme `/ɽ/` is not contained in any phoneme tables of Phoible, and needs to be corrected to `/ʃ/` after listening |
| [`ɯ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɯ.ogg) | [adayl`ı`klar](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/adaylıklar.mp3) | a d a j ɫ `ɯ` k ɫ a ɾ |  |
| [`ʋ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʋ.ogg) | [kri`v`okapiç](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/krivokapiç.mp3) | k ɾ i `ʋ` o k a p i t͡ʃ |  |
| [`ɫ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɫ.ogg) | [aday`l`ık`l`ar](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/adaylıklar.mp3) | a d a j `ɫ` ɯ k `ɫ` a ɾ | The phoneme `/ɫ/` is not contained in [SPA 186](https://phoible.org/inventories/view/186), but contained in [EA 2416](https://phoible.org/inventories/view/2416) |
| [`a`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/a.ogg) | [`a`celemiz](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/acelemiz.mp3) | `a` d͡ʒ e l e m i z |  |
| [`b`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/b.ogg) | [`b`agiç'e](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/bagiç'e.mp3) | `b` a ɟ i t͡ʃ e |  |
| [`c`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/k.ogg) | [an`k`ete](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/ankete.mp3) | a n `c` e t e | Incorrect G2P labeling. The letter `k` is pronounced as `/k/`, so the phoneme `/c/` needs to be corrected to `/k/` |
| [`d`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/d.ogg) | [a`d`aylıklar](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/adaylıklar.mp3) | a `d` a j ɫ ɯ k ɫ a ɾ |  |
| [`dʒ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/dʒ.ogg) | [a`c`elemiz](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/acelemiz.mp3) | a `d͡ʒ` e l e m i z | The same pronunciation for `/dʒ/` and `/d͡ʒ/`, so replacing `/dʒ/` with `/d͡ʒ/` |
| [`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/e.ogg) | [ac`e`lemiz](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/acelemiz.mp3) | a d͡ʒ `e` l e m i z | The phoneme `/e/` is not contained in [SPA 186](https://phoible.org/inventories/view/186), but contained in [UZ 2217](https://phoible.org/inventories/view/2217) |
| [`ɛ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɛ.ogg) | [m`e`xico](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/mexico.mp3) | m e `ɛ` k s i d͡ʒ o | Incorrect G2P labeling. The phoneme `/e/` is redundant and the phoneme `/d͡ʒ/` need to be corrected to `/k/` after listening |
| [`f`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/f.ogg) | [a`ff`edebilecek](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/affedebilecek.mp3) | a `f` e d e b i l e c e k |  |
| [`h`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/h.ogg) | [`h`arekât](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/hareka%CC%82t.mp3) | `h` a ɾ e k a t |  |
| [`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/i.ogg) | [acelem`i`z](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/acelemiz.mp3) | a d͡ʒ e l e m `i` z |  |
| [`j`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/j.ogg) | [ada`y`lıklar](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/adaylıklar.mp3) | a d a `j` ɫ ɯ k ɫ a ɾ |  |
| [`k`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/k.ogg) | [adaylı`k`lar](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/adaylıklar.mp3) | a d a j ɫ ɯ `k` ɫ a ɾ |  |
| [`l`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/l.ogg) | [ace`l`emiz](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/acelemiz.mp3) | a d͡ʒ e `l` e m i z | The letter `l` is pronounced as `/l/` before letter `e`, `i`, `ö` or `ü`, otherwise it is pronounced as `/ɫ/` |
| [`m`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/m.ogg) | [acele`m`iz](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/acelemiz.mp3) | a d͡ʒ e l e `m` i z |  |
| [`n`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/n.ogg) | [alevle`n`iverdi](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/alevleniverdi.mp3) | a l e ʋ l e `n` i w e r d i |  |
| [`ŋ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ŋ.ogg) | [a`n`kara'yı](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/ankara'yı.mp3) | a `ŋ` k a r a j ɨ | Incorrect G2P labeling. The phoneme `/ŋ/` is not contained in any phoneme tables of LanguageNet or Phoible, and needs to be corrected to `/n/` after listening |
| [`o`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/o.ogg) | [görülmüy`o`rdu](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/görülmüyordu.mp3) | ɡ ø r y l m y j `o` r d u |  |
| [`ø`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ø.ogg) | [g`ö`rülmüyordu](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/görülmüyordu.mp3) | ɡ `ø` r y l m y j o r d u | The phoneme `/ø/` is not contained in [SPA 186](https://phoible.org/inventories/view/186), but contained in [EA 2416](https://phoible.org/inventories/view/2416) |
| [`p`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/p.ogg) | [krivoka`p`iç](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/krivokapiç.mp3) | k ɾ i ʋ o k a `p` i t͡ʃ |  |
| [`q`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/q.ogg) | [fran`q`uelin](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/franquelin.mp3) | f ɾ a n `q` u e l i n | Incorrect G2P labeling. The phoneme `/q/` is not contained in any phoneme tables of LanguageNet or Phoible. The phoneme `/q/` needs to be corrected to `/k/` after listening |
| [`r`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/r.ogg) | [dijitalleşti`r`ilmiştir](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/dijitalleştirilmiştir.mp3) | d i ʒ i t a ɫ l e ʃ t i `r` i l m i ʃ t i ɽ |  |
| [`s`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/s.ogg) | [abartıyor`s`un](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/abartıyorsun.mp3) | a b a r t ɨ j o n | The G2P labeling misses phoneme `/s/`, and needs to be corrected to `/a b a r t ɯ j o s u n/` |
| [`t`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/t.ogg) | [diji`t`alleştirilmiştir](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/dijitalleştirilmiştir.mp3) | d i `ʒ` i `t` a ɫ l e ʃ t i r i l m i ʃ t i ɽ |  |
| [`tʃ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/tʃ.ogg) | [krivokapi`ç`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/krivokapiç.mp3) | k ɾ i ʋ o k a p i `t͡ʃ` | The same pronunciation for `/tʃ/` and `/t͡ʃ/`, so replacing `/tʃ/` with `/t͡ʃ/` |
| [`u`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/u.ogg) | [abartıyors`u`n](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/abartıyorsun.mp3) | a b a r t ɨ j o n | The G2P labeling misses phoneme `/u/`, and needs to be corrected to `/a b a r t ɯ j o s u n/` |
| [`v`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/v.ogg) | [`v`etëvendosje](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/vete%CC%88vendosje.mp3) | `v` e t e v e n d o s ʒ e | The G2P labeling misses phoneme `/u/`, and needs to be corrected to `/a b a r t ɯ j o s u n/` |
| [`w`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʋ.ogg) | [alevleni`v`erdi](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/alevleniverdi.mp3) | a l e ʋ l e n i `w` e r d i | Incorrect G2P labeling. The phoneme `/w/` is not contained in any phoneme tables of LanguageNet or Phoible, and needs to be corrected to `/ʋ/` |
| [`x`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/k.ogg) | [synta`x`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/syntax.mp3) | s j n t a `x` | The letter `x` is pronounced as `/k s/`, and the phoneme `/x/` is not contained in any phoneme tables of LanguageNet or Phoible. So the phoneme `/x/` needs to be corrected to `/k s/` |
| [`y`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/y.ogg) | [gör`ü`lm`ü`yordu](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/görülmüyordu.mp3) | ɡ ø r `y` l m `y` j o r d u |  |
| [`z`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/z.ogg) | [acelemi`z`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/acelemiz.mp3) | a d͡ʒ e l e m i `z` |  |
| [`ʒ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʒ.ogg) | [di`j`italleştirilmiştir](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/tr/word_audio/dijitalleştirilmiştir.mp3) | d i `ʒ` i t a ɫ l e ʃ t i r i l m i ʃ t i ɽ |  |
