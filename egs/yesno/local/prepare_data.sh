#!/usr/bin/env bash
# This script prepares data and create necessary files

. ./path.sh

data=${H}/data
local=${H}/local 
mkdir -p ${data}/local

cd ${data}

# acquire data if not downloaded
if [ ! -d waves_yesno ]; then
  echo "Getting Data"
  wget http://www.openslr.org/resources/1/waves_yesno.tar.gz || exit 1;
  tar -xvzf waves_yesno.tar.gz || exit 1;
  rm waves_yesno.tar.gz || exit 1;
fi

echo "Preparing train and dev data"

rm -rf train dev

# Create waves list and Divide into dev and train set
waves_dir=${data}/waves_yesno
ls -1 $waves_dir | grep "wav" > ${data}/local/waves_all.list
cd ${data}/local
${local}/create_yesno_waves_test_train.pl waves_all.list waves.dev waves.train
#rm waves.dev | cp waves.train waves.dev

cd ${data}/local

for x in train dev; do
  # create id lists
  ${local}/create_yesno_wav_scp.pl ${waves_dir} waves.$x > ${x}_wav.scp #id to wavfile
  ${local}/create_yesno_txt.pl waves.$x > ${x}.txt #id to content
done

${local}/create_yesno_wav_scp.pl ${waves_dir} waves.dev > test_wav.scp #id to wavfile
${local}/create_yesno_txt.pl waves.dev > test.txt #id to content

cd ${data}

for x in train dev test; do
  # sort wave lists and create utt2spk, spk2utt
  mkdir -p $x
  sort local/${x}_wav.scp -o $x/wav.scp
  sort local/$x.txt -o $x/text
  cat $x/text | awk '{printf("%s global\n", $1);}' > $x/utt2spk
  sort $x/utt2spk -o $x/utt2spk
  ${H}/utils/utt2spk_to_spk2utt.pl < $x/utt2spk > $x/spk2utt
done

#cp ${input}/task.arpabo lm_tg.arpa