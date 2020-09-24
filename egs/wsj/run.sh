#!/bin/bash

# Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
# Apache 2.0.
# This script implements CTC-CRF training on WSJ dataset.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
stage=1
wsj0=/data/csr_1
wsj1=/data/csr_2_comp

. utils/parse_options.sh

if [ $stage -le 1 ]; then
  echo "Data Preparation and FST Construction"
  # Use the same datap prepatation script from Kaldi
  local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;

  # # Construct the phoneme-based lexicon from the CMU dict
  local/wsj_prepare_phn_dict.sh || exit 1;
  ctc-crf/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;

  local/wsj_extend_dict.sh --dict-suffix "_phn" $wsj1/13-32.1
  local/wsj_train_lms_phn.sh --dict-suffix "_phn"
  # Compile the lexicon and token FSTs
  ctc-crf/ctc_compile_dict_token.sh data/local/dict_phn_larger data/local/lang_phn_larger_tmp data/lang_phn_larger || exit 1;

  # Compile the language-model FST and the final decoding graph TLG.fst
  local/wsj_format_local_lms.sh --lang-suffix "_phn"
  local/wsj_decode_graph.sh data/lang_phn || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "FBank Feature Generation"
  # Split the whole training data into training (95%) and cross-validation (5%) sets
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train_si284 data/train_tr95 data/train_cv05 || exit 1
  utils/data/perturb_data_dir_speed_3way.sh data/train_tr95 data/train_tr95_sp
  utils/data/perturb_data_dir_speed_3way.sh data/train_cv05 data/train_cv05_sp
  # echo " preparing directory for speed-perturbed data done"

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  fbankdir=fbank
  for set in train_tr95_sp train_cv05_sp; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 20 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
  done

  for set in test_dev93 test_eval92; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 20 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;
  done
fi

data_tr=data/train_tr95_sp
data_cv=data/train_cv05_sp

if [ $stage -le 3 ]; then
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_tr/text "<UNK>" > $data_tr/text_number || exit 1
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_cv/text "<UNK>" > $data_cv/text_number || exit 1
  echo "convert text_number finished"

  # prepare denominator

  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_tr95/text "<UNK>" > data/train_tr95/text_number
  cat data/train_tr95/text_number | sort -k 2 | uniq -f 1 > data/train_tr95/unique_text_number
  mkdir -p data/den_meta
  chain-est-phone-lm ark:data/train_tr95/unique_text_number data/den_meta/phone_lm.fst || exit 1
  ctc-crf/ctc_token_fst_corrected.py den data/lang_phn/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst
  fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst
  echo "prepare denominator finished"

  path_weight $data_tr/text_number data/den_meta/phone_lm.fst > $data_tr/weight
  path_weight $data_cv/text_number data/den_meta/phone_lm.fst > $data_cv/weight
  echo "prepare weight finished"
fi

if [ $stage -le 4 ]; then
  feats_tr="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$data_tr/feats.scp ark:- \
    | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
  feats_cv="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$data_cv/feats.scp ark:- \
    | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
  mkdir -p data/tmp
  copy-feats "$feats_tr" "ark,scp:data/tmp/tr.ark,data/tmp/tr.scp"
  copy-feats "$feats_cv" "ark,scp:data/tmp/cv.ark,data/tmp/cv.scp"

  mkdir -p data/hdf5
  python ctc-crf/convert_to_hdf5.py data/tmp/cv.scp $data_cv/text_number $data_cv/weight data/hdf5/cv.hdf5 || exit 1
  python ctc-crf/convert_to_hdf5.py data/tmp/tr.scp $data_tr/text_number $data_tr/weight data/hdf5/tr.hdf5 || exit 1
fi

data_dev93=data/test_dev93
data_eval92=data/test_eval92

if [ $stage -le 5 ]; then
  feats_dev93="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_dev93/utt2spk scp:$data_dev93/cmvn.scp scp:$data_dev93/feats.scp ark:- \
    | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
  feats_eval92="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_eval92/utt2spk scp:$data_eval92/cmvn.scp scp:$data_eval92/feats.scp ark:- \
    | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
  mkdir -p data/test_data
  copy-feats "$feats_dev93" "ark,scp:data/test_data/dev93.ark,data/test_data/dev93.scp"
  copy-feats "$feats_eval92" "ark,scp:data/test_data/eval92.ark,data/test_data/eval92.scp"
fi

arch=BLSTM
dir=exp/$arch

if [ $stage -le 6 ]; then
  python3 ctc-crf/train.py \
    --min_epoch=8 \
    --output_unit=72 \
    --arch=$arch \
    --lamb=0.01 \
    --data_path=data/hdf5 \
    $dir
fi

if [ $stage -le 7 ]; then
 for set in dev93 eval92; do
     ark_dir=$dir/logits/${set}
     mkdir -p $ark_dir
     CUDA_VISIBLE_DEVICES=0 python3 ctc-crf/calculate_logits.py \
       --nj=20 --input_scp=data/test_data/${set}.scp \
       --config=$dir/config.json --model=$dir/best_model --output_dir=$ark_dir
 done
fi

if [ $stage -le 8 ]; then
  nj=20
  for set in dev93 eval92; do
     for lmtype in tgpr bd_tgpr; do
        # reuse logits
        mkdir -p $dir/decode_${set}_${lmtype}
        ln -s $(readlink -f $dir/logits/${set}) $dir/decode_${set}_${lmtype}/logits
        ctc-crf/decode.sh --stage 1 \
          --cmd "$decode_cmd" --nj $nj --acwt 1.0 \
          data/lang_phn_test_${lmtype} data/test_${set} data/test_data/${set}.scp $dir/decode_${set}_${lmtype}
     done
  done


  for set in dev93 eval92; do
    mkdir -p $dir/decode_${set}_bd_fgconst
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_phn_test_bd_{tgpr,fgconst} data/test_${set} $dir/decode_${set}_bd_{tgpr,fgconst} || exit 1;
    mkdir -p $dir/exp/decode_${set}_tg
    steps/lmrescore.sh --cmd "$decode_cmd" --mode 3 data/lang_phn_test_{tgpr,tg} data/test_${set} $dir/decode_${set}_{tgpr,tg} || exit 1;
  done
fi
