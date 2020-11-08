#!/bin/bash

# Copyright 2020 Keyu An ankeyuthu@gmail.com

. ./cmd.sh

. path.sh

stage=1
. utils/parse_options.sh
dir=`pwd -P`
downsample=true

if [ $stage -le 1 ]; then
  echo ===========================================================
  echo "          Data Preparation and FST Construction          "
  echo ===========================================================
  # Use the same datap preparation script from Kaldi
  local/hkust_data_prep.sh /data/LDC2005S15/ /data/LDC2005T32/  || exit 1;
  # Run the original script for dict preparation
  local/hkust_prepare_dict.sh || exit 1;

  # Construct the phoneme-based dict
  # We get 118 tokens, representing phonemes with tonality
  local/hkust_prepare_phn_dict.sh || exit 1;

  # Compile the lexicon and token FSTs
  ctc-crf/ctc_compile_dict_token.sh --dict-type "phn" \
    data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;
  # Train and compile LMs
  local/hkust_train_lms.sh data/local/train/text data/local/dict_phn/lexicon.txt data/local/lm_phn || exit 1;
  
  # Compile the language-model FST and the final decoding graph TLG.fst
  local/hkust_decode_graph.sh data/local/lm_phn data/lang_phn data/lang_phn_test || exit 1;
fi

if [ $stage -le 2 ]; then
  echo ===========================================================
  echo "               FBank Feature Generation                  "
  echo ===========================================================
  fbankdir=fbank
  # Use the first 4k sentences as dev, around 5 hours
  utils/subset_data_dir.sh --first data/train 4000 data/train_dev
  n=$[`cat data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last data/train $n data/train_nodev
  # prepare directory for speed-perturbed data
  utils/data/perturb_data_dir_speed_3way.sh data/train_nodev data/train_nodev_sp
  utils/data/perturb_data_dir_speed_3way.sh data/train_dev data/train_dev_sp
  echo "prepareing directory for speed-perturbed data done"

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  for set in train_nodev_sp train_dev_sp; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 20 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
  done

  for set in dev; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 20 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
  done
fi

data_tr=data/train_nodev_sp
data_cv=data/train_dev_sp

if [ $stage -le 3 ]; then
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_tr/text "<UNK>" > $data_tr/text_number || exit 1
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_cv/text "<UNK>" > $data_cv/text_number || exit 1
  echo "convert text_number finished"

  # prepare denominator

  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_nodev/text "<UNK>" > data/train_nodev/text_number
  cat data/train_nodev/text_number | sort -k 2 | uniq -f 1 > data/train_nodev/unique_text_number
  mkdir -p data/den_meta
  chain-est-phone-lm ark:data/train_nodev/unique_text_number data/den_meta/phone_lm.fst
  ctc-crf/ctc_token_fst_corrected.py den data/lang_phn/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst
  fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst
  echo "prepare denominator finished"

  path_weight $data_tr/text_number data/den_meta/phone_lm.fst > $data_tr/weight || exit 1
  path_weight $data_cv/text_number data/den_meta/phone_lm.fst > $data_cv/weight || exit 1
  echo "prepare weight finished"
fi

if [ $stage -le 4 ]; then
  if [ "$downsample" == "true" ]; then
    feats_tr="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$data_tr/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
    feats_cv="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$data_cv/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
  else
    feats_tr="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$data_tr/feats.scp ark:- \
     | add-deltas ark:- ark:- |"
    feats_cv="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$data_cv/feats.scp ark:- \
     | add-deltas ark:- ark:- |"
  fi 

  ark_dir=data/all_ark
 
  mkdir -p $ark_dir
  copy-feats "$feats_tr" "ark,scp:$ark_dir/tr.ark,$ark_dir/tr.scp" || exit 1
  copy-feats "$feats_cv" "ark,scp:$ark_dir/cv.ark,$ark_dir/cv.scp" || exit 1  
  echo "copy -feats finished"

  mkdir -p data/pkl
  python ctc-crf/convert_to_pickle.py $ark_dir/cv.scp $data_cv/text_number $data_cv/weight data/pkl/cv.pkl || exit 1
  python ctc-crf/convert_to_pickle.py $ark_dir/tr.scp $data_tr/text_number $data_tr/weight data/pkl/tr.pkl || exit 1
fi

data_test=data/dev

if [ $stage -le 5 ]; then
  if [ "$downsample" == "true" ]; then
    feats_test="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_test/utt2spk scp:$data_test/cmvn.scp scp:$data_test/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
  else
    feats_test="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_test/utt2spk scp:$data_test/cmvn.scp scp:$data_test/feats.scp ark:- \
      | add-deltas ark:- ark:- |"
  fi
  mkdir -p data/dev_data
  copy-feats "$feats_test" "ark,scp:data/dev_data/test.ark,data/dev_data/test.scp"
fi

arch=BLSTM
dir=exp/$arch

if [ $stage -le 6 ]; then
  echo "nn training."
  python ctc-crf/train.py \
    --min_epoch=5 \
    --arch=$arch \
    --output_unit=119 \
    --lamb=0.01 \
    --batch_size=128 \
    --pkl \
    --data_path=data/pkl \
    $dir || exit 1
fi

if [ $stage -le 7 ]; then
  for set in dev; do
    CUDA_VISIBLE_DEVICES=0 \
    ctc-crf/decode.sh --stage 0 --cmd "$decode_cmd" --nj 20 --acwt 1.0 \
      data/lang_phn_test data/$set data/${set}_data/test.scp $dir/decode
  done
fi
   
