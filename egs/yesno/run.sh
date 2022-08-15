#!/bin/bash

# Copyright 2018-2021 Tsinghua University
# Author: Siwei Li
# yesno for CAT

# environment
. ./cmd.sh
. ./path.sh

if [ ! -d $H ]; then
  echo "Home directory not set yet." && exit 1;
fi

#set
n=12     # parallel jobs=$(nproc)
stage=6  # set work stages
stop_stage=9
change_config=1

. utils/parse_options.sh

NODE=$1
if [ ! $NODE ]; then
  NODE=0
fi

if [ $NODE == 0 ]; then
  if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "stage 1: Data Preparation and FST Construction"

    local/prepare_data.sh || exit 1; # Get data and lists
    local/prepare_dict.sh || exit 1; # Get lexicon dict

    # Compile the lexicon and token FSTs
    # generate lexicon FST L.fst according to words.txt, generate token FST T.fst according to tokens.txt
    ctc-crf/ctc_compile_dict_token.sh --dict-type "phn" \
      data/dict data/local/lang_phn_tmp data/lang || exit 1;

    # Train and compile LMs. Generate G.fst according to lm, and compose FSTs into TLG.fst
    local/yesno_train_lms.sh data/train/text data/dict/lexicon.txt data/lm || exit 1;
    local/yesno_decode_graph.sh data/lm/srilm/srilm.o1g.kn.gz data/lang data/lang_test || exit 1;
  fi

  if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "stage 2: FBank Feature Generation"
    #perturb the speaking speed to achieve data augmentation
    utils/data/perturb_data_dir_speed_3way.sh data/train data/train_sp
    utils/data/perturb_data_dir_speed_3way.sh data/dev data/dev_sp
    
    # Generate the fbank features; by default 40-dimensional fbanks on each frame
    fbankdir=fbank
    for set in train_sp dev_sp; do
      steps/make_fbank.sh --cmd "$train_cmd" --nj 1 data/$set exp/make_fbank/$set $fbankdir || exit 1;
      utils/fix_data_dir.sh data/$set || exit;  #filter and sort the data files
      steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;  #achieve cmvn normalization
    done

    for set in test; do
      steps/make_fbank.sh --cmd "$train_cmd" --nj 1 data/$set exp/make_fbank/$set $fbankdir || exit 1;
      utils/fix_data_dir.sh data/$set || exit;  #filter and sort the data files
      steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;  #achieve cmvn normalization
    done
  fi

  data_tr=data/train_sp
  data_cv=data/dev_sp

  if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    #convert word sequences to label sequences according to lexicon_numbers.txt and text files in data/lang_phn
    #the result will be placed in $data_tr/ and $data_cv/
    ctc-crf/prep_ctc_trans.py data/lang/lexicon_numbers.txt $data_tr/text "<UNK>" > $data_tr/text_number
    ctc-crf/prep_ctc_trans.py data/lang/lexicon_numbers.txt $data_cv/text "<UNK>" > $data_cv/text_number
    echo "convert text_number finished"

    # prepare denominator
    ctc-crf/prep_ctc_trans.py data/lang/lexicon_numbers.txt data/train/text "<UNK>" > data/train/text_number
    #sort the text_number file, and then remove the duplicate lines
    cat data/train/text_number | sort -k 2 | uniq -f 1 > data/train/unique_text_number
    mkdir -p data/den_meta
    #generate phone_lm.fst, a phone-based language model
    chain-est-phone-lm ark:data/train/unique_text_number data/den_meta/phone_lm.fst
    #generate the correct T.fst, called T_den.fst
    ctc-crf/ctc_token_fst_corrected.py den data/lang/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst
    #compose T_den.fst and phone_lm.fst into den_lm.fst
    fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst
    echo "prepare denominator finished"
    
    #calculate and save the weight for each label sequence based on text_number and phone_lm.fst
    path_weight $data_tr/text_number data/den_meta/phone_lm.fst > $data_tr/weight
    path_weight $data_cv/text_number data/den_meta/phone_lm.fst > $data_cv/weight
    echo "prepare weight finished"
  fi

  if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    mkdir -p data/all_ark
    
    for set in test; do
      eval data_$set=data/$set
    done

    for set in test cv tr; do
      tmp_data=`eval echo '$'data_$set`

      #apply CMVN feature normalization, calculate delta features, then sub-sample the input feature sequence
      feats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$tmp_data/utt2spk scp:$tmp_data/cmvn.scp scp:$tmp_data/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"

      ark_dir=$(readlink -f data/all_ark)/$set.ark
      #copy feature files, generate scp and ark files to save features.
      copy-feats "$feats" "ark,scp:$ark_dir,data/all_ark/$set.scp" || exit 1
    done
  fi

  if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    mkdir -p data/pickle
    #create a pickle file to save the feature, text_number and path weights.
    python3 ctc-crf/convert_to.py -f=pickle --describe='L//4' --filer=1500 \
        data/all_ark/cv.scp $data_cv/text_number $data_cv/weight data/pickle/cv.pickle || exit 1
    python3 ctc-crf/convert_to.py -f=pickle --describe='L//4' --filer=1500 \
        data/all_ark/tr.scp $data_tr/text_number $data_tr/weight data/pickle/tr.pickle || exit 1
  fi
fi

PARENTDIR='.'
dir="exp/demo"
DATAPATH=$PARENTDIR/data/

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then

  if [ $change_config == 1 ]; then
    rm $dir/scripts.tar.gz
    rm -rf $dir/ckpt
  fi

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
  CUDA_VISIBLE_DEVICES="0"                      \
  python3 ctc-crf/train.py --seed=0             \
    --world-size 1 --rank $NODE                 \
    --batch_size=3                              \
    --dir=$dir                                  \
    --config=$dir/config.json                   \
    --data=$DATAPATH                            \
    || exit 1
fi

if [ $NODE -ne 0 ]; then
  exit 0
fi

# nj=$(nproc)
nj=1
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  for set in test; do
    ark_dir=$dir/logits/$set
    mkdir -p $ark_dir
    python3 ctc-crf/calculate_logits.py               \
      --resume=$dir/ckpt/bestckpt.pt                     \
      --config=$dir/config.json                       \
      --nj=$nj --input_scp=data/all_ark/$set.scp      \
      --output_dir=$ark_dir                           \
      || exit 1
  done
fi


if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  for set in test; do
    mkdir -p $dir/decode_${set}
    ln -s $(readlink -f $dir/logits/$set) $dir/decode_${set}/logits
    ctc-crf/decode.sh --stage 1 \
        --cmd "$decode_cmd" --nj 1 --acwt 1.0 --post_decode_acwt 1.0\
        data/lang_${set} data/${set} data/all_ark/${set}.scp $dir/decode_${set}
  done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  for set in test; do
    grep WER $dir/decode_${set}/wer_* | utils/best_wer.sh
  done
fi
