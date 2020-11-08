#!/bin/bash

# Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
# Apache 2.0.
# This script implements CTC-CRF training on LibriSpeech dataset.

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11
stage=1

. ./cmd.sh
. ./path.sh

data=/data/librispeech
lm_src_dir=/data/librispeech_lm

. ./cmd.sh
. ./path.sh

if [ $stage -le 0 ]; then
   for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
     local/download_and_untar.sh $data $data_url $part
   done
   # download the LM resources
   local/download_lm.sh $lm_url data/local/lm
fi

if [ $stage -le 1 ]; then
  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
  # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done

  fbankdir=fbank
  for part in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_fbank/$part $fbankdir
    steps/compute_cmvn_stats.sh data/$part exp/make_fbank/$part $fbankdir
  done

  # ... and then combine the two sets into a 460 hour one
  utils/combine_data.sh \
    data/train_clean_460 data/train_clean_100 data/train_clean_360

  # combine all the data
  utils/combine_data.sh \
    data/train_960 data/train_clean_460 data/train_other_500
fi

if [ $stage -le 2 ]; then
  # copy the LM resources
  #local/copy_lm.sh $lm_src_dir data/local/lm
  local/prepare_dict_ctc.sh data/local/lm data/local/dict_phn
  ctc-crf/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;
  local/format_lms.sh --src-dir data/lang_phn data/local/lm

  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  echo "3" > data/lang_phn/oov.int
  utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
    data/lang_phn data/lang_phn_tglarge
  utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
    data/lang_phn data/lang_phn_fglarge

  langdir=data/lang_phn_tgsmall
  fsttablecompose ${langdir}/L.fst $langdir/G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstarcsort --sort_type=ilabel > $langdir/LG.fst || exit 1;
  fsttablecompose ${langdir}/T.fst $langdir/LG.fst > $langdir/TLG.fst || exit 1;

fi

data_tr=data/train_tr95
data_cv=data/train_cv05

if [ $stage -le 3 ]; then
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train_960  data/train_tr95 data/train_cv05 || exit 1

  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_tr/text "<UNK>" > $data_tr/text_number
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_cv/text "<UNK>" > $data_cv/text_number
  echo "convert text_number finished"

  # prepare denominator
  cat $data_tr/text_number | sort -k 2 | uniq -f 1 > $data_tr/unique_text_number
  mkdir -p data/den_meta
  chain-est-phone-lm ark:$data_tr/unique_text_number data/den_meta/phone_lm.fst
  utils/ctc_token_fst_corrected.py den data/lang_phn/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst
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
  mkdir -p data/all_ark
  copy-feats "$feats_tr" "ark,scp:data/all_ark/tr.ark,data/all_ark/tr.scp"
  copy-feats "$feats_cv" "ark,scp:data/all_ark/cv.ark,data/all_ark/cv.scp"

  mkdir -p data/hdf5
  python3 ctc-crf/convert_to_hdf5.py data/all_ark/cv.scp $data_cv/text_number $data_cv/weight data/hdf5/cv.hdf5
  python3 ctc-crf/convert_to_hdf5.py data/all_ark/tr.scp $data_tr/text_number $data_tr/weight data/hdf5/tr.hdf5
fi

arch=BLSTM
dir=exp/$arch
output_unit=$(awk '{if ($1 == "#0")print $2 - 1 ;}' data/lang_phn/tokens.txt)

if [ $stage -le 5 ]; then
    echo "nn training."
    python3 ctc-crf/train.py \
    --arch=$arch \
    --output_unit=$output_unit \
    --lamb=0.1 \
    --data_path \
    data/hdf5 \
    $dir
fi

graphdir=data/lang_phn_tgsmall

if [ $stage -le 6 ]; then
  for set in test_clean test_other dev_clean dev_other; do
    data_test=data/$set
    feats_test="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_test/utt2spk scp:$data_test/cmvn.scp scp:$data_test/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
    copy-feats "$feats_test" "ark,scp:data/all_ark/${set}.ark,data/all_ark/${set}.scp"

    CUDA_VISIBLE_DEVICES=0 \
    ctc-crf/decode.sh --cmd "$decode_cmd" --nj 20 --acwt 1.0 \
      data/lang_phn_test $data_test data/all_ark/${set}.scp $dir/decode_${set}
  done
fi


if [ $stage -le 7 ]; then
  for set in test_clean test_other dev_clean dev_other; do
    nj=20
    lm=tgmed
    rescore_dir=$dir/decode_${set}/lattice_${lm}
    mkdir -p $rescore_dir
    ./local/lmrescore.sh --cmd "$train_cmd" data/lang_phn_{tgsmall,$lm} data/$set $dir/decode_${set}/lattice_{tgsmall,$lm} $nj || exit 1;

    for lm in tglarge fglarge; do
      rescore_dir=$dir/decode_${set}/lattice_${lm}
      mkdir -p $rescore_dir
      ./local/lmrescore_const_arpa.sh --cmd "$train_cmd" data/lang_phn_{tgsmall,$lm} data/$set $dir/decode_${set}/lattice_{tgsmall,$lm} $nj || exit 1;
    done
  done
fi
