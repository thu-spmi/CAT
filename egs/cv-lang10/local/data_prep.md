# Data preprocessing

Follow the 5 steps to process data for a given language：
* Data Downloading
* Kaldi Format Data Generation 
* Excluding dev and test from the training set
* Text Normalization
* Word List Generation
* Pronunciation Lexicon Generation
* Feature Extraction

We will explain in detail for Polish as below:

```bash
export LC_ALL=C.UTF-8
stage=1
stop_stage=6
lang=pl
download_url="https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-11.0-2022-09-21/cv-corpus-11.0-2022-09-21-${lang}.tar.gz"  # URL for CommonVoice languages dataset. The URL may change, it is recommended to get the download link on https://commonvoice.mozilla.org/en/datasets
work_space=CAT/egs/cv-lang10  # your working path
dict_dir=$work_space/dict/$lang
data_dir=$work_space/data/$lang
wav_dir=$work_space/data/cv-corpus-11.0-2022-09-21/$lang
kaldi_root=/opt/kaldi
```

## Stage -1: Data Downloading

Download data from [CommonVoice official website](https://commonvoice.mozilla.org/zh-CN/datasets) and unzip it

```bash
wget -c $download_url -O cv-corpus-${lang}.tar.gz
tar -xvf cv-corpus-${lang}.tar.gz 
```

## Stage 0: Kaldi Format Data Generation 

Extract audio paths and transcription texts from the unzipped csv files to generate Kaldi format data. Only `text` and `wav.scp` files need to be generated.

```bash
cd $work_space
mkdir -p $data_dir
for s in dev test train validated;do
    d_set="$data_dir/$s"
    mkdir -p $d_set
    file="$wav_dir/$s.tsv"
          [ ! -f $file ] && {
              echo "No such file $file"
              exit 1
          }
    cut <$file -f 2 | tail -n +2 | xargs basename -s ".mp3" >$d_set/uid.tmp
    cut <$file -f 2 | tail -n +2 | awk -v path="$wav_dir/clips" '{print path"/"$1}' >$d_set/path.tmp
    paste $d_set/{uid,path}.tmp | sort -k 1,1 -u >$d_set/wav.scp
    cut <$file -f 3 | tail -n +2 >$d_set/text.tmp
    paste $d_set/{uid,text}.tmp | sort -k 1,1 -u >$d_set/text
    rm -rf $d_set/{uid,text,path}.tmp
done
```

The variable `language_list` in the code is the list of language IDs to be processed, `d_set` is the path where the processed data is stored, and `file` is the path of the original transcription text; This stage mainly realizes the generation of `text` and `wav.scp` based on audio data and paths.

The data format in `text` is as follows:
```
$ head -2 data/pl/dev/text
common_voice_pl_100540	same way you did
common_voice_pl_10091129	by hook or by crook
```

The data format in `wav.scp` is as follows:
```
$ head -2 data/pl/dev/text
common_voice_pl_20551620	/mnt/workspace/CAT/egs/commonvoice/data/cv-corpus-11.0-2022-09-21/pl/clips/common_voice_pl_20551620.mp3
common_voice_pl_20594755	/mnt/workspace/CAT/egs/commonvoice/data/cv-corpus-11.0-2022-09-21/pl/clips/common_voice_pl_20594755.mp3
```

## Stage 1: Excluding dev and test from the training set
```bash
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # By default, I use validated+train as the real training data
  # ... but we must exclude the dev & test from the validated one.
  echo "stage 1: Exclude the dev & test from the train set"
  d_train="$data_dir/excluded_train"
  mkdir -p $d_train
  for file in wav.scp text; do
    cat $data_dir/{validated,train}/$file |
        sort -k 1,1 -u >$d_train/$file.tmp
    for exc_set in dev test; do
        python local/expect.py \
          $d_train/$file.tmp \
          --exclude $data_dir/$exc_set/$file \
          >$d_train/$file.tmp.tmp
        mv $d_train/$file.tmp.tmp $d_train/$file.tmp
    done
    mv $d_train/$file.tmp $d_train/$file
  done
  rm -rf $data_dir/{validated,train}
  echo $lang 'Text done'
fi
```
## Stage 2: Text normalization

Remove special characters such as punctuation marks and foreign language characters from the transcription `text`, as non-linguistic symbols cannot be recognized by the G2P model to generate a pronunciation lexicon.
```bash
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
  echo "stage 2: Text Normalization"
  # Text Normalization
  for set in dev test excluded_train; do
    	paste $data_dir/$set/text > $data_dir/$set/text.bak
    	cut <$data_dir/$set/text.bak -f 2- | \
        sed -e 's/`/ /g; s/¨/ /g; s/~/ /g; s/=/ /g' \
         -e 's/|/ /g; s/°/ /g; s/[-]/ /g; s/[―]/ /g; s/,/ /g; s/[;]/ /g; s/:/ /g; s/!/ /g; s/¡/ /g; s/?/ /g; s/[¿]/ /g; s/′/ /g; s/‐/ /g; s/´´/ /g' \
         -e 's/[.]/ /g; s/·/ /g; s/‘/ /g; s/’/ /g; s/"/ /g; s/“/ /g; s/”/ /g; s/«/ /g; s/»/ /g; s/≪/ /g; s/≫/ /g; s/[{]/ /g; s/„/ /g; s/−/ /g; s/‑/ /g' \
         -e 's/[}]/ /g; s/®/ /g; s/→/ /g; s/ʿ/ /g; s/‧/ /g; s/ʻ/ /g; s/ ⃗/ /g; s/‹/ /g; s/›/ /g; s/_/ /g; s/ʽ//g; s/￼￼/ /g; s/m̪/m/g; s/ː/ /g; s/ﬁ/fi/g; s/ﬂ/fl/g' \
         -e 's/[–]/ /g; s/…/ /g' \
         -e "s/\// /g; s/#/ /g; s/&/ & /g; s/´/'/g; s/''/ /g; s/^[']*/ /g; s/[']*$/ /g; s/ '/ /g; s/' / /g; s/\[/ /g; s/\]/ /g" \
         -e 's/&/ /g;s/(/ /g;s/)/ /g;s/\\/ /g;s/—/ /g;s/，/ /g;s/！/ /g;' | \
         sed -e 's/[ ][ ]*/ /g; s/^[ ]*//g; s/[ ]*$//g' | \
         python -c "import sys; print(sys.stdin.read().lower())" > data/$lang/$set/text.trans.tmp
      	cut <$data_dir/$set/text.bak -f 1 > $data_dir/$set/text.id.tmp
        paste $data_dir/$set/text.{id,trans}.tmp > $data_dir/$set/text
        cat $data_dir/$set/text | sed -e 's/^[	]*//g' | grep -v "^$" > $data_dir/$set/text_new
        mv $data_dir/$set/text_new $data_dir/$set/text
        rm -rf $data_dir/$set/text.{id,trans}.tmp
    done
  echo $lang 'Text normalization done'
fi
```

Each language’s text contains different special symbols. Therefore, it is necessary to print out all the characters from the text, then we input the characters into the G2P model and count the unrecognized characters. And run `text_norm.sh` to remove them.

Note:

* For non-linguistic symbols, we add the characters to be removed in the script and use the script to remove them. Note that some characters like `$`, `\`, etc., which need to be preceded by an escape character `\`, otherwise the command cannot be executed correctly.
* For foreign characters, since they contain pronunciation information, simply deleting a word may affect model training. Therefore, it is necessary to delete the entire sentence containing foreign characters.
* To avoid accidental deletion, it is recommended to back up `text` and `wav.scp` before deletion. All normalization operations should only be performed on `text`.

## Stage 3: Word List Generation

Count the words in the transcription text and generate a unique word list, with one word per line, formatted as follows:
```
$ head -3 $dict_dir/word_list
a
aaron
ababa
```

```bash
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
	mkdir -p $dict_dir
  	text_file="$data_dir/*/text"
  	cat $text_file | awk -F '\t' '{print $NF}'  | sed -e 's| |\n|g' | grep -v "^$" | sort -u -s > $dict_dir/word_list
  	echo $lang 'Word list done'
  	python local/char_list.py $dict_dir/word_list
  	echo $lang 'character list done, please check special tokens in character list, confirm Text normalization is correct.'
fi
```
Where `text_file` is the path of the transcription.

## Stage 4: Pronunciation Lexicon Generation

Firstly, it is necessary to download the G2P model from https://github.com/uiuc-sst/g2ps. The we use the transcription text after normalization as the input of the G2P model to generate the pronunciation lexicon. Each line in the generated lexicon is a mapping from word to phoneme, formatted as follows:
```
$ head -3 $dict_dir/lexicon.txt
a	ə
aaron	a a r o n
ababa	a b ə b ə
```

```bash
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
  # Generating lexicon and lexicon correction
  echo "stage 4: G2P Conversion, generating lexicon"
  g2ps=g2ps/models 
  phonetisaurus-apply --model $g2ps/polish_2_2_2.fst --word_list dict/pl_mls/word_list > dict/pl_mls/lexicon.txt
  cat dict/pl_mls/lexicon.txt | awk '{$1=""; print $0}' | sed -e 's/ts/t͡s/g; s/dz/d͡z/g; s/ɖʐ/ɖ͡ʐ/g; s/tʂ/ʈ͡ʂ/g; s/dʑ/d͡ʑ/g; s/tɕ/t͡ɕ/g; s/ɔ̃/ɔ/g; s/ɨ̃/ɨ/g; s/ɛ̃/ɛ/g; s/w̃/w/g; s/ɛ̝/ɛ/g; s/s̪/s/g; s/n̪/n/g; s/t̪/t/g; s/z̪/z/g' > dict/pl_mls/phone.txt
  sed -i 's/ː//g; s/ˈ//g; s/ʲ//g; s/[ ][ ]*/ /g; s/^[ ]*//g; s/[ ]*$//g' dict/$lang/phone.txt
  cat dict/$lang/lexicon.txt | awk '{print $1}' > dict/$lang/word.txt
  paste dict/$lang/{word,phone}.txt > dict/$lang/lexicon_new.txt
  mv dict/$lang/lexicon_new.txt dict/$lang/lexicon.txt
  rm -rf dict/$lang/{lexicon_new,word,phone}.txt
  echo $lang 'Lexicon done'
fi
```

Since the G2P model used has a certain error rate, it may generate incorrect symbols. Therefore, after generating the lexicon, post-processing is required, such as removing extra symbols and splitting diphonemes into monophoneme. The specific processing for each language needs to refer to the corresponding LanguageNet symbol table in http://www.isle.illinois.edu/speech_web_lg/data/g2ps/

## Stage 5-6: Feature Extraction
Extract fbank feature from audio data using kaldi toolkit.
```bash
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Get duration from mp3 file, use kaldi toolkit"
    for ti in dev test excluded_train;do
      # generate utt2spk and spk2utt that kaldi needed
      awk '{print $1,$1}' $data_dir/${ti}/wav.scp > $data_dir/${ti}/utt2spk
      cp $data_dir/${ti}/utt2spk $data_dir/${ti}/spk2utt

      # add ffmpeg command for wav.scp file
      mv $data_dir/$ti/wav.scp $data_dir/$ti/wav_mp3.scp
      awk '{print $1 "\tffmpeg -i " $2 " -f wav -ar 16000 -ab 16 -ac 1 - |"}' $data_dir/$ti/wav_mp3.scp > $data_dir/$ti/wav.scp
      
      # Get duration
      cd $kaldi_root/egs/wsj/s5
      utils/data/get_utt2dur.sh $data_dir/$ti

      # Get total duration
      python local/tools/calculate_dur.py $data_dir/$ti
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "stage 6: make F-bank feature, use kaldi toolkit"
    for ti in dev test excluded_train;do
      # fix data
      cd $kaldi_root/egs/wsj/s5
      utils/fix_data_dir.sh $data_dir/$ti

      mkdir -p $data_dir/$ti/conf
      echo "--num-mel-bins=80" > $data_dir/$ti/conf/fbank.conf
      steps/make_fbank.sh --fbank-config $data_dir/$ti/conf \
                          $data_dir/$ti \
                          $data_dir/$ti/log \
                          $data_dir/$ti/fbank
    done
fi
```
