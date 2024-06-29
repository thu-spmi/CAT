# Italian
Author: Ma, Te (mate153125@gmail.com)
## 1. Text normalization 

(1) The G2P models cannot recognize alien words, so we choose to remove sentences that containing alien words. They are listed in the file [`Italian_alien_sentences.txt`](/home/mate/cat_multilingual/egs/cv-lang10/data-process/it/Italian_alien_sentences.txt).

(2) Before creating lexicon, we need to normalize text. The code of text normalization for __Italian__ is in the script named [`text_norm.sh`](./text_norm.sh).

## 2. Lexicon generation and correction

We use the FST (Finite State Transducer) based G2P (Grapheme-to-Phoneme) toolkit, [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus), to create the pronunciation lexicon. The trained FSTs for use with Phonetisaurus is provided in [LanguageNet](https://github.com/uiuc-sst/g2ps#languagenet-grapheme-to-phoneme-transducers).

Note that the above G2P procedure is not perfect. As noted in `LanguageNet`, "PERs range from 7% to 45%".
The G2P-generated lexicon needs to be corrected. The correction step is based on [the LanguageNet symbol table for __Italian__](https://github.com/uiuc-sst/g2ps/blob/masterItalian/Italian_wikipedia_symboltable.html). The code of this step of lexicon correction is in the script named [`lexicon.sh`](./lexicon.sh).

(1) We remove some special symbols such as accent symbols to enable sharing more phonemes between different languages.

| Removed symbols | Note |
| ------ | ------ |
| `ː` | Accent | 
| `ˈ` | Long vowel |
| `ˌ` | Syllable |

(2) For the phoneme `/t͡ʃ/`, we need to revise `/tʃ/` to a single phoneme `/t͡ʃ/`. So as for `/d͡ʒ/` and `/d͡z/`. A further subtle issue is that IPA symbols may be encoded in different forms. So to enforce consistency, the phoneme `/∅/` is corrected to `/ø/`.

| Phoneme from G2P | Phoneme corrected |
| ------ | ------ |
| `tʃ` | `t͡ʃ` |
| `dʒ` | `d͡ʒ` |
| `dz` | `d͡z` |
| `ts` | `t͡s` |
| `∅` | `ø` |

## 3. Check of phonemes

Strictly speaking, one phoneme might correspond to multiple phones (those phones are referred to as the allophones). Note that our above procedure removes the diacritic, the notion of phonemes in this work is a looser one.

The generated lexicon from the G2P procedure is named [`lexicon_it.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/lexicon_it.txt). The set of IPA phonemes appeared in the lexicon is saved in [`phone_list.txt`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/phone_list.txt). We further check `phone_list.txt`, by referring to the following two phoneme lists and with listening tests.  

* IPA symbol table in LanguageNet, which, thought by LanguageNet, contains all the phones in the language:
https://github.com/uiuc-sst/g2ps/blob/masterItalian/Italian_wikipedia_symboltable.html
  
* IPA symbol table in Phoible: 
https://phoible.org/languages/ital1282. For each language, there may exist multiple phoneme inventories, which are archived at the Phoible website. 
For __Italian__, we choose the first one as the main reference for phoneme checking, which is [PH 1145](https://phoible.org/inventories/view/1145).

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

| IPA symbol in `phone_list.txt` | Word |  <div style="width: 170pt">G2P labeling result | Note |
| ------ | ------ | ------ | ------ |
| [`ɲ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɲ.mp3) | [arma`gn`acchi](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/armagnacchi.mp3) | a r m a u `ɲ` `ɲ` a k k j | Incorrect G2P labeling. The redundant phonemes `/u ɲ/` should be removed |
| [`ʎ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʎ.mp3) | ['heimskrin`gl`a](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/'heimskringla.mp3) | ø e i m s k r i n `ʎ` a |  |
| [`ʃ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ʃ.mp3) | [caravagge`sc`o](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/caravaggesco.mp3) | k a r a v a u ɡ d͡ʒ e `ʃ` o |  |
| [`ɡ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɡ.mp3) | [carava`gg`esco](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/caravaggesco.mp3) | k a r a v a u `ɡ` d͡ʒ e ʃ o |  |
| [`a`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/a.mp3) | [annu`a`rio](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/'annuario.mp3) | a n n w `a` r i o | The phoneme `/a/` is not contained in [PH 1145](https://phoible.org/inventories/view/1145), but contained in [UZ 2195](https://phoible.org/inventories/view/2195) |
| [`b`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/b.mp3) | [pu`b`blicizzò](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/pubblicizzo%CC%80.mp3) | p `b` b l i t͡ʃ t͡s d͡z ɔ |  |
| [`d`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/d.mp3) | [quest'affermazione](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/quest'affermazione.mp3) | k u e s t a f f ɛ n `d` r m a t͡s i o n e | Incorrect G2P labeling. Redundant phonemes `/n d/` should be removed |
| [`dz`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/dz.mp3) | [pubbliciz`z`ò](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/pubblicizzo%CC%80.mp3) | p b b l i t͡ʃ t͡s `d͡z` ɔ | The same pronunciation for `/dz/` and `/d͡z/`, so replacing `/dz/` with `/d͡z/` |
| [`dʒ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/dʒ.mp3) | [caravag`g`esco](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/caravaggesco.mp3) | k a r a v a u ɡ `d͡ʒ` e ʃ o | The same pronunciation for `/dʒ/` and `/d͡ʒ/`, so replacing `/dʒ/` with `/d͡ʒ/` |
| [`e`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/e.mp3) | [qu`e`st'affermazione](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/quest'affermazione.mp3) | k u `e` s t a f f ɛ n d r m a t͡s i o n e |  |
| [`ɛ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɛ.mp3) | [quest'aff`e`rmazione](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/quest'affermazione.mp3) | k u e s t a f f `ɛ` n d r m a t͡s i o n e |  |
| [`f`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/f.mp3) | [quest'a`ff`ermazione](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/quest'affermazione.mp3) | k u e s t a `f` `f` ɛ n d r m a t͡s i o n e |  |
| [`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/i.mp3) | [pubbl`i`cizzò](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/pubblicizzo%CC%80.mp3) | p b b l `i` t͡ʃ t͡s d͡z ɔ |  |
| [`j`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/j.mp3) | [armagnacch`i`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/armagnacchi.mp3) | a r m a u ɲ ɲ a k k `j` |  |
| [`k`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/k.mp3) | [`q`uest'affermazione](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/quest'affermazione.mp3) | `k` u e s t a f f ɛ n d r m a t͡s i o n e |  |
| [`l`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/l.mp3) | [pubb`l`icizzò](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/pubblicizzo%CC%80.mp3) | p b b `l` i t͡ʃ t͡s d͡z ɔ |  |
| [`m`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/m.mp3) | [quest'affer`m`azione](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/quest'affermazione.mp3) | k u e s t a f f ɛ n d r `m` a t͡s i o n e |  |
| [`n`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/n.mp3) | [an`n`uario](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/annuario.mp3) | a n `n` w a r i o |  |
| [`o`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/o.mp3) | [annuari`o`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/annuario.mp3) | a n n w a r i `o` |  |
| [`ø`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ø.mp3) | [`h`eimskringla](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/heimskringla.mp3) | `ø` e i m s k r i n ʎ a | The phoneme `/ø/` is not contained in any phoneme tables of Phoible, and the letter `h` seems to be silent, so the redundant phoneme `/ø/` should be removed |
| [`ɔ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ɔ.mp3) | [pubblicizz`ò`](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/pubblicizzo%CC%80.mp3) | p b b l i t͡ʃ t͡s d͡z `ɔ` |  |
| [`p`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/p.mp3) | [`p`ubblicizzò](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/pubblicizzo%CC%80.mp3) | `p` b b l i t͡ʃ t͡s d͡z ɔ |  |
| [`r`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/r.mp3) | [annua`r`io](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/annuario.mp3) | a n n w a `r` i o |  |
| [`s`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/s.mp3) | [que`s`t'affermazione](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/quest'affermazione.mp3) | k u e `s` t a f f ɛ n d r m a t͡s i o n e |  |
| [`t`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/t.mp3) | [ques`t`'affermazione](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/quest'affermazione.mp3) | k u e s `t` a f f ɛ n d r m a t͡s i o n e |  |
| [`tʃ`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/tʃ.mp3) | [pubbli`ci`zzò](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/pubblicizzo%CC%80.mp3) | p b b l i `t͡ʃ` t͡s d͡z ɔ | The same pronunciation for `/tʃ/` and `/t͡ʃ/`, so replacing `/tʃ/` with `/t͡ʃ/` |
| [`ts`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/ts.mp3) | [pubblici`z`zò](http://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/pubblicizzo%CC%80.mp3) | p b b l i t͡ʃ `t͡s` d͡z ɔ | The same pronunciation for `/ts/` and `/t͡s/`, so replacing `/ts/` with `/t͡s/` |
| [`u`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/u.mp3) | [q`u`est'affermazione](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/quest'affermazione.mp3) | k `u` e s t a f f ɛ n d r m a t͡s i o n e |  |
| [`v`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/v.mp3) | [cara`v`aggesco](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/caravaggesco.mp3) | k a r a `v` a u ɡ d͡ʒ e ʃ o |  |
| [`w`](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/IPA_audio/w.mp3) | [ann`u`ario](https://cat-ckpt.oss-cn-beijing.aliyuncs.com/cat-multilingual/cv-lang10/dict/it/word_audio/annuario.mp3) | a n n `w` a r i o |  |
