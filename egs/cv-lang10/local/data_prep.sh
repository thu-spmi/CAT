# Copyright 2023 Tsinghua SPMI Lab, Author: Mate (1531253797@qq.com)
# Acknowlegement: This script refer to the code of Huahuan Zheng (maxwellzh@outlook.com)
# This script is used to prepare data and lexicon for each language.

# set -x -u
export LC_ALL=C.UTF-8
stage=1
stop_stage=6
lang=$1
download_url="https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-11.0-2022-09-21/cv-corpus-11.0-2022-09-21-${lang}.tar.gz"  # URL for CommonVoice languages dataset. The URL may change, it is recommended to get the download link on https://commonvoice.mozilla.org/en/datasets
work_space=CAT/egs/cv-lang10  # your working path
dict_dir=$work_space/dict/$lang
data_dir=$work_space/data/$lang
wav_dir=$work_space/data/cv-corpus-11.0-2022-09-21/$lang
kaldi_root=/opt/kaldi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
  #Download and unzip data
  cd $work_space/data
  wget -c $download_url -O cv-corpus-${lang}.tar.gz
  tar -xvf cv-corpus-${lang}.tar.gz 
  cd -
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
  # Extract meta info
  echo "stage 0: Prepare data from tsv file"
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
fi


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


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
  echo "stage 2: Text Normalization"
  # Text Normalization
  bash $data_dir/text_norm.sh $data_dir
  echo $lang 'Text normalization done'
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
  echo "stage 3: Generating word_list"
  # Generating word_list
  mkdir -p $dict_dir
  text_file="$data_dir/*/text"
  cat $text_file | awk -F '\t' '{print $NF}'  | sed -e 's| |\n|g' | grep -v "^$" | sort -u -s > $dict_dir/word_list
  echo $lang 'Word list done'
  python local/tools/char_list.py $dict_dir/word_list
  echo $lang 'character list done, please check special tokens in character list, confirm Text normalization is correct.'
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
  # Generating lexicon and lexicon correction
  echo "stage 4: G2P Conversion, generating lexicon"
  bash $data_dir/lexicon.sh $dict_dir
  sed -i 's/ː//g; s/ˈ//g; s/ʲ//g; s/[ ][ ]*/ /g; s/^[ ]*//g; s/[ ]*$//g' $dict_dir/phone.txt
  cat $dict_dir/lexicon.txt | awk '{print $1}' > $dict_dir/word.txt
  paste $dict_dir/{word,phone}.txt > $dict_dir/lexicon_new.txt
  mv $dict_dir/lexicon_new.txt $dict_dir/lexicon.txt
  rm -rf $dict_dir/{lexicon_new,word,phone}.txt
  echo $lang 'Lexicon done'
fi

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
