#!/usr/bin/env bash
#
# Copyright 2018, Yuan-Fu Liao, National Taipei University of Technology, yfliao@mail.ntut.edu.tw
#           2020  AsusTek Computer Inc. (Author: Alex Hung)
# Before you run this recipe, please apply, download and put or make a link of the corpus under this folder (folder name: "NER-Trs-Vol1").
# For more detail, please check:
# 1. Formosa Speech in the Wild (FSW) project (https://sites.google.com/speech.ntut.edu.tw/fsw/home/corpus)
# 2. Formosa Speech Recognition Challenge (FSW) 2018 (https://sites.google.com/speech.ntut.edu.tw/fsw/home/challenge)
stage=-3
num_jobs=20

train_dir=NER-Trs-Vol1/Train
test_dir=NER-Trs-Vol1-Test
test_key_dir=NER-Trs-Vol1-Test-Key
eval_dir=NER-Trs-Vol1-Eval
eval_key_dir=NER-Trs-Vol1-Eval-Key

# shell options
set -eo pipefail

. ./cmd.sh
. ./path.sh
. parse_options.sh

# configure number of jobs running in parallel, you should adjust these numbers according to your machines
# data preparation
if [ $stage -le -3 ]; then
  echo "$0: Data Preparation"
  local/prepare_data.sh || exit 1;

  echo "$0: Lexicon Preparation"
  local/prepare_dict.sh || exit 1;

  echo "$0: Phone Sets, questions, L compilation Preparation"
  rm -rf data/lang
  utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
      "<SIL>" data/local/lang data/lang || exit 1;

  echo "$0: Prepare phone dictionary"
  rm -rf data/lang_phn
  local/prepare_phn_dict.sh || exit 1;

  echo "$0: Compile the lexicon and token FSTs"
  ctc-crf/ctc_compile_dict_token.sh --dict-type "phn" \
    data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;

  echo "$0: LM training"
  rm -rf data/local/lm/3gram-mincount
  local/train_lms.sh || exit 1;

  echo "$0: Compile the language-model FST and the final decoding graph TLG.fst"
  local/format_lm.sh data/local/lm data/lang_phn data/lang_phn_test || exit 1;
fi

if [ $stage -le -2 ]; then
  echo "$0: making mfccs"
  for x in train; do
    steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $num_jobs data/$x || exit 1;
    steps/compute_cmvn_stats.sh data/$x || exit 1;
    utils/fix_data_dir.sh data/$x || exit 1;
  done
  echo "$0: train mono model"
  utils/subset_data_dir.sh --shortest data/train 3000 data/train_mono

  steps/train_mono.sh --boost-silence 1.25 --cmd "$train_cmd" --nj $num_jobs \
    data/train_mono data/lang exp/mono || exit 1;

  # Get alignments from monophone system.
  steps/align_si.sh --boost-silence 1.25 --cmd "$train_cmd" --nj $num_jobs \
    data/train data/lang exp/mono exp/mono_ali || exit 1;

  echo "$0: train tri1 model"
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
   2500 20000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

  # align tri1
  steps/align_si.sh --cmd "$train_cmd" --nj $num_jobs \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1;

  echo "$-: train tri2 model"
  # Train tri3a, which is LDA+MLLT,
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
   2500 20000 data/train data/lang exp/tri1_ali exp/tri2a || exit 1;

  steps/cleanup/segment_long_utterances.sh --cmd "$train_cmd" \
    --num-neighbors-to-search 0 --nj $num_jobs \
    exp/tri2a data/lang data/train data/train_reseg \
    exp/segment_long_utts_train || exit 1;

  utils/fix_data_dir.sh data/train_reseg || exit 1;
fi

if [ $stage -le -1 ]; then
  utils/data/perturb_data_dir_speed_3way.sh data/train_reseg data/train_sp || exit 1;
  echo "$0: making hires-mfcc"
  for x in train_sp test eval; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $num_jobs --mfcc-config conf/mfcc_hires.conf data/$x || exit 1;
    steps/compute_cmvn_stats.sh data/$x || exit 1;
    utils/fix_data_dir.sh data/$x || exit 1;
  done
fi

data_tr=data/train_sp
data_cv=data/test
if [ $stage -le 0 ]; then
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_tr/text "<SIL>" > $data_tr/text_number
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_cv/text "<SIL>" > $data_cv/text_number
  echo "convert text_number finished"

  # prepare denominator
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train/text "<SIL>" > data/train/text_number
  cat data/train/text_number | sort -k 2 | uniq -f 1 > data/train/unique_text_number
  mkdir -p data/den_meta
  chain-est-phone-lm ark:data/train/unique_text_number data/den_meta/phone_lm.fst
  ctc-crf/ctc_token_fst_corrected.py den data/lang_phn/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst
  fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst
  echo "prepare denominator finished"
fi

ark_dir=data/all_ark
if [ $stage -le 1 ]; then
  path_weight $data_tr/text_number data/den_meta/phone_lm.fst > $data_tr/weight
  path_weight $data_cv/text_number data/den_meta/phone_lm.fst > $data_cv/weight
  echo "prepare weight finished"

  feats_tr="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$data_tr/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
  feats_cv="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$data_cv/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"

  mkdir -p $ark_dir
  copy-feats "$feats_tr" "ark,scp:$ark_dir/tr.ark,$ark_dir/tr.scp" || exit 1
  copy-feats "$feats_cv" "ark,scp:$ark_dir/cv.ark,$ark_dir/cv.scp" || exit 1

  echo "copy -feats finished"
  mkdir -p data/hdf5

  python3 ctc-crf/convert_to_hdf5.py $ark_dir/cv.scp $data_cv/text_number $data_cv/weight data/hdf5/cv.hdf5
  python3 ctc-crf/convert_to_hdf5.py $ark_dir/tr.scp $data_tr/text_number $data_tr/weight data/hdf5/tr.hdf5
fi

if [ $stage -le 2 ]; then
  feats_test="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:data/test/utt2spk scp:data/test/cmvn.scp scp:data/test/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
  feats_eval="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:data/eval/utt2spk scp:data/eval/cmvn.scp scp:data/eval/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
  ark_dir=data/all_ark
  copy-feats "$feats_test" "ark,scp:$ark_dir/test.ark,$ark_dir/test.scp" || exit 1
  copy-feats "$feats_eval" "ark,scp:$ark_dir/eval.ark,$ark_dir/eval.scp" || exit 1
fi

dir=exp/tdnnlstm
output_unit=$(awk '{if ($1 == "#0")print $2 - 1 ;}' data/lang_phn/tokens.txt)

if [ $stage -le 3 ]; then
    echo "nn training."
    # takes 12gb ram on gpu and 23gb ram on mainboard.
    python3 ctc-crf/train.py \
      --lr=0.00001 \
      --hdim=1024 \
      --layers=3 \
      --min_epoch=1 \
      --arch=TDNN_LSTM \
      --batch_size=20 \
      --output_unit=$output_unit \
      --lamb=1.0 \
      --data_path=data/hdf5 \
      $dir
fi

if [ $stage -le 4 ]; then
  CUDA_VISIBLE_DEVICES=0 \
  ctc-crf/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj $num_jobs \
    data/lang_phn_test data/test $ark_dir/test.scp $dir/decode_test
  CUDA_VISIBLE_DEVICES=0 \
  ctc-crf/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj $num_jobs \
    data/lang_phn_test data/eval $ark_dir/eval.scp $dir/decode_eval
fi

# getting results (see RESULTS file)
if [ $stage -le 5 ]; then
  echo "$0: extract the results"
  for test_set in test eval; do
  echo "WER: $test_set"
  for x in exp/*/decode_*${test_set}*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
  for x in exp/*/*/decode_*${test_set}*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
  echo

  echo "CER: $test_set"
  for x in exp/*/decode_*${test_set}*; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
  for x in exp/*/*/decode_*${test_set}*; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
  echo
  done
fi

# finish
echo "$0: all done"

exit 0;
