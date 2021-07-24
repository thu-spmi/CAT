#!/bin/bash

# Copyright 2018-2021 Tsinghua University
# Author: Hongyu Xiang, Huahuan Zheng
# Apache 2.0.
# This script implements CTC-CRF training on SwitchBoard dataset.
# It's updated to v2 by Huahuan Zheng in 2021, based on CAT branch v1 egs/wsj/run.sh

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh

stage=3
stop_stage=100
data=/mnt/workspace/pengwenjie/espnet/egs/commonvoice/asr1/download/de_data/cv-corpus-5.1-2020-06-22/de
data_url=de.tar.gz
lang=de
train_set=train_$(echo $lang | tr - _)
dev=dev_$(echo $lang | tr - _)
test_set=test_$(echo $lang | tr - _)
recog_set="${dev} $test_set"
nbpe=150
bpemode=unigram


NODE=$1
if [ ! $NODE ]; then
  NODE=0
fi

if [ $NODE == 0 ]; then
  if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Data Preparation and FST Construction"
    # Use the same data preparation script from Kaldi
    #local/download_and_untar.sh $(dirname $data) $data_url
    for part in "validated" "test" "dev"; do
        # use underscore-separated names in data directories.
        local/data_prep.pl "${data}" ${part} data/"$(echo "${part}_${lang}" | tr - _)" || exit 1;
    done
#
#    # remove test&dev data from validated sentences
    utils/copy_data_dir.sh data/"$(echo "validated_${lang}" | tr - _)" data/${train_set}
    utils/filter_scp.pl --exclude data/dev_de/wav.scp data/train_de/wav.scp > data/train_de/temp_wav.scp
    utils/filter_scp.pl --exclude data/test_de/wav.scp data/train_de/temp_wav.scp > data/train_de/wav.scp
    utils/fix_data_dir.sh data/train_de


    utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1
    utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/${train_set} data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/${train_set} data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/train_de data/temp1 data/temp2 data/temp3
    rm -rf data/{temp1,temp2,temmp3}

  fi

  if [ $stage -le 2 ] && [ $stage -ge 1 ]; then

    fbankdir=fbank
    for x in train dev test; do
      mkdir -p data/$x
      cp -r data/${x}_de/* data/$x/
      steps/make_fbank.sh --cmd "$train_cmd" --nj 20 data/${x} exp/make_fbank/$x \
          $fbankdir || exit 1;
      utils/fix_data_dir.sh data/${x} || exit 1;
      steps/compute_cmvn_stats.sh data/${x} exp/make_fbank/$x $fbankdir || exit 1
    done

    # remove_longshortdata.sh needs feats.scp
    for x in train dev test; do
        # Remove features with too long frames in training data
      max_len=3000
      local/remove_longshortdata.sh  --maxframes $max_len data/${x}_de data/$x
      cp -r data/${x}_de/* data/$x/
    done    
  fi


  if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then

    echo "stage 3: Dictionary and Json Data Preparation"
    local/cv_prepare_bpe_dict.sh || exit 1

    ctc-crf/ctc_compile_dict_token.sh --dict_type "bpe" data/local/dict_bpe \
        data/local/lang_bpe_tmp data/lang_bpe || exit 1;
    echo "Building n-gram LM model."
    
    # train.txt without uttid for training n-gramm
    cat data/train/text_pos | cut -f 2- -d " " - > data/local/dict_bpe/train.txt
    local/cv_train_lm.sh data/local/dict_bpe/train.txt data/local/dict_bpe/ data/local/local_lm || exit 1;
    local/cv_format_local_lms.sh --lang-suffix "bpe" 


    for x in train dev; do
        ctc-crf/prep_ctc_trans.py data/lang_bpe/lexicon_numbers.txt data/$x/text_pos \
            "<SPN>" > data/${x}/text_id_number || exit 1;
    done
    echo "convert text_number finished."

    cat data/train/text_id_number | sort -k 2 | uniq -f 1 > data/train/unique_text_id_number
    mkdir -p data/den_meta
    chain-est-phone-lm ark:data/train/unique_text_id_number data/den_meta/phone_lm.fst || exit 1;
    ctc-crf/ctc_token_fst_corrected.py den data/lang_bpe/tokens.txt | fstcompile \
        | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst
    fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst
    echo "prepare denominator finished"
    path_weight data/train/text_id_number data/den_meta/phone_lm.fst > data/train/weight
    path_weight data/dev/text_id_number data/den_meta/phone_lm.fst > data/dev/weight
    echo "prepare weight finished"

    exit 0
  fi 
    
  if [ $stage -le 4 ] && [ ${stop_stage} -ge 4 ]; then

    feats_train="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:data/train/utt2spk scp:data/train/cmvn.scp scp:data/train/feats.scp ark:- \
        | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
    feats_dev="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:data/dev/utt2spk scp:data/dev/cmvn.scp scp:data/dev/feats.scp ark:- \
        | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
    feats_test="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:data/test/utt2spk scp:data/test/cmvn.scp scp:data/test/feats.scp ark:- \
        | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"

    mkdir -p data/tmp
    copy-feats "$feats_train" "ark,scp:data/tmp/train.ark,data/tmp/train.scp"
    copy-feats "$feats_dev" "ark,scp:data/tmp/dev.ark,data/tmp/dev.scp"
    copy-feats "$feats_test" "ark,scp:data/tmp/test.ark,data/tmp/test.scp"

    mkdir -p data/hdf5
    python ctc-crf/convert_to_pickle.py data/tmp/train.scp data/train/text_id_number \
        data/train/weight data/hdf5/train.pkl || exit 1;
    python ctc-crf/convert_to_pickle.py data/tmp/dev.scp data/dev/text_id_number \
        data/dev/weight data/hdf5/dev.pkl || exit 1;
  fi

  data_eval2000=data/eval2000
  ark_dir=exp/decode_eval2000/ark

  if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    feats_eval2000="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_eval2000/utt2spk scp:$data_eval2000/cmvn.scp scp:$data_eval2000/feats.scp ark:- \
         | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"

    mkdir data/test_data
    copy-feats "$feats_eval2000" ark,scp:data/test_data/eval2000.ark,data/test_data/eval2000.scp || exit 1;
  fi
fi

exit 0
PARENTDIR='.'
dir="exp/demo"
DATAPATH=$PARENTDIR/data/

if [ $stage -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  unset CUDA_VISIBLE_DEVICES

  if [[ $NODE == 0 && ! -f $dir/scripts.tar.gz ]]; then
    echo ""
    tar -zcf $dir/scripts.tar.gz $(readlink ctc-crf) $0
  elif [ $NODE == 0 ]; then
    echo ""
    echo "'$dir/scripts.tar.gz' already exists."
    echo "If you want to update it, please manually rm it then re-run this script."
  fi

  # uncomment the following line if you want to use specified GPUs
  # CUDA_VISIBLE_DEVICES="0"                    \
  python3 ctc-crf/train.py --seed=0             \
    --world-size 1 --rank $NODE                 \
    --batch_size=128                            \
    --dir=$dir                                  \
    --config=$dir/config.json                   \
    --data=$DATAPATH                            \
    || exit 1
fi

if [ $NODE -ne 0 ]; then
  exit 0
fi


nj=$(nproc)
if [ $stage -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  for set in eval2000; do
    ark_dir=$dir/logits/${set}
    mkdir -p $ark_dir
    CUDA_VISIBLE_DEVICES=0                            \
    python3 ctc-crf/calculate_logits.py               \
      --resume=$dir/ckpt/infer.pt                     \
      --config=$dir/config.json                       \
      --nj=$nj --input_scp=data/test_data/${set}.scp  \
      --output_dir=$ark_dir                           \
      || exit 1
 done
fi

if [ $stage -le 8 ] && [ ${stop_stage} -ge 8 ]; then

  for x in test; do
    for lmtype in bd_tgpr bd_fgpr; do
      des=$dir/decode_${x}_${lmtype}
      logits=$dir/logits/$x
      mkdir -p $des
      ctc-crf/decode.sh --stage 1 --cmd "$decode_cmd" --nj $nj --acwt 1.0 \
          data/lang_bpe_test_$lmtype data/${x} $logits data/tmp/${x}.scp $des 
    done
  done

  for x in test; do
    mkdir -p $dir/decode_${x}_bd_fgconst
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_bpe_test_bd_{tgpr,fgconst} \
        data/$x $dir/decode_${x}_bd_{tgpr,fgconst} || exit 1;
    mkdir -p $dir/exp/decode_${x}_bd_tg
    steps/lmrescore.sh --cmd "$decode_cmd" --mode 3 data/lang_bpe_test_bd_{tgpr,tg} data/$x \
        $dir/decode_${x}_bd_{tgpr,tg} || exit 1
  done

fi

exit 0
