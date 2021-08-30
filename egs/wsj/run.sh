#!/bin/bash

# Copyright 2018-2021 Tsinghua University
# Author: Hongyu Xiang, Huahuan Zheng
# Apache 2.0.
# This script implements CTC-CRF training on WSJ dataset.
# It's updated to v2 by Huahuan Zheng in 2021, based on CAT branch v1 egs/wsj/run.sh

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh
stage=1
stop_stage=100
wsj0=/data/csr_1
wsj1=/data/csr_2_comp

. utils/parse_options.sh

NODE=$1
if [ ! $NODE ]; then
    NODE=0
fi

if [ $NODE == 0 ]; then
  if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Data Preparation and FST Construction"
    # Use the same datap prepatation script from Kaldi
    local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1;

    # # Construct the phoneme-based lexicon from the CMU dict
    local/wsj_prepare_phn_dict.sh || exit 1;
    ctc-crf/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;

    local/wsj_extend_dict.sh --dict-suffix "_phn" $wsj1/13-32.1 || exit 1
    local/wsj_train_lms_phn.sh --dict-suffix "_phn" || exit 1
    # Compile the lexicon and token FSTs
    ctc-crf/ctc_compile_dict_token.sh data/local/dict_phn_larger data/local/lang_phn_larger_tmp data/lang_phn_larger || exit 1;

    # Compile the language-model FST and the final decoding graph TLG.fst
    local/wsj_format_local_lms.sh --lang-suffix "_phn" || exit 1
    local/wsj_decode_graph.sh data/lang_phn || exit 1;
  fi

  if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "FBank Feature Generation"
    # Split the whole training data into training (95%) and cross-validation (5%) sets
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 data/train_si284 data/train_tr95 data/train_cv05 || exit 1
    utils/data/perturb_data_dir_speed_3way.sh data/train_tr95 data/train_tr95_sp || exit 1
    utils/data/perturb_data_dir_speed_3way.sh data/train_cv05 data/train_cv05_sp || exit 1
    echo "Preparing directory for speed-perturbed data done"

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

  if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    python3 ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_tr/text "<UNK>" > $data_tr/text_number || exit 1
    python3 ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_cv/text "<UNK>" > $data_cv/text_number || exit 1
    echo "Convert text_number finished"

    # Prepare denominator
    python3 ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_tr95/text "<UNK>" > data/train_tr95/text_number || exit 1
    cat data/train_tr95/text_number | sort -k 2 | uniq -f 1 > data/train_tr95/unique_text_number || exit 1
    mkdir -p data/den_meta
    chain-est-phone-lm ark:data/train_tr95/unique_text_number data/den_meta/phone_lm.fst || exit 1
    python3 ctc-crf/ctc_token_fst_corrected.py den data/lang_phn/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst || exit 1
    fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst || exit 1
    echo "Prepare denominator finished"
  
    # For label sequence l, log p(l) also appears in the numerator but behaves like an constant. So log p(l) is
    # pre-calculated based on the denominator n-gram LM and saved, and then applied in training.
    path_weight $data_tr/text_number data/den_meta/phone_lm.fst > $data_tr/weight || exit 1
    path_weight $data_cv/text_number data/den_meta/phone_lm.fst > $data_cv/weight || exit 1
    echo "Prepare weight finished"
  fi

  if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    mkdir -p data/all_ark
    for set in dev93 eval92; do
      eval data_$set=data/test_$set
    done
    for set in dev93 eval92 cv tr; do
      tmp_data=`eval echo '$'data_$set`
      feats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$tmp_data/utt2spk scp:$tmp_data/cmvn.scp scp:$tmp_data/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"

      ark_dir=$(readlink -f data/all_ark)/$set.ark
      copy-feats "$feats" "ark,scp:$ark_dir,data/all_ark/$set.scp" || exit 1
    done
  fi

  if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    mkdir -p data/pickle
    python3 ctc-crf/convert_to.py -f=pickle -W \
        data/all_ark/cv.scp $data_cv/text_number $data_cv/weight data/pickle/cv.pickle || exit 1
    python3 ctc-crf/convert_to.py -f=pickle \
        data/all_ark/tr.scp $data_tr/text_number $data_tr/weight data/pickle/tr.pickle || exit 1
  fi

fi

PARENTDIR='.'
dir="exp/demo"
DATAPATH=$PARENTDIR/data/
########################################################################
# For multi-nodes training,                                            #
# assure numbers of available GPUs in each node are the same           #
#                                                                      #
# Add the following arguments to python3 ctc-crf/train.py              #
# '--dist-url="tcp://<IP of node 0>:<port>"'                           #
# And change the '--world-size' to number of nodes                     #
#                                                                      #
# In node 0: execute                                                   #
# $ ./run.sh 0                                                         #
# In node 1: execute                                                   #
# $ ./run.sh 1                                                         #
# ...                                                                  #
# NOTE: the '--dist-url' MUST be the same across nodes and binding to  #
# node 0 address.                                                      #
# '--rank' is the index of node (No need of manual modification).      #
########################################################################
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
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
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  for set in dev93 eval92; do
    ark_dir=$dir/logits/$set
    mkdir -p $ark_dir
    python3 ctc-crf/calculate_logits.py             \
      --resume=$dir/ckpt/bestckpt.pt                \
      --config=$dir/config.json                     \
      --nj=$nj                                      \
      --input_scp=data/all_ark/$set.scp             \
      --output_dir=$ark_dir                         \
      || exit 1

      for lmtype in bd_tgpr; do
        # reuse logits
        mkdir -p $dir/decode_${set}_$lmtype
        ln -snf $(readlink -f $dir/logits/$set) $dir/decode_${set}_$lmtype/logits
        ctc-crf/decode.sh --stage 1 \
          --cmd "$decode_cmd" --nj $nj --acwt 1.0 \
          data/lang_phn_test_$lmtype data/test_${set} data/all_ark/$set.scp $dir/decode_${set}_$lmtype
      done

    mkdir -p $dir/decode_${set}_bd_fgconst
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_phn_test_bd_{tgpr,fgconst} data/test_$set $dir/decode_${set}_bd_{tgpr,fgconst} || exit 1;
  done
fi

grep WER $dir/decode_eval92_*/wer_* | utils/best_wer.sh
grep WER $dir/decode_dev93_*/wer_* | utils/best_wer.sh
