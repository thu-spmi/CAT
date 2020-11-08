. ./cmd.sh
. ./path.sh

stage=1
H=`pwd`  #exp home
n=8      #parallel jobs#!/usr/bin/env bash

thchs=/mnt/nas_workspace2/spmiData/THCHS30  #Data Path

if [ $stage -le 1 ]; then
  echo "Data Preparation and FST Construction"
  # Use the same datap prepatation script from Kaldi, create directory data/train, data/dev and data/test, 
  # create spk2utt, utt2dur and other files in each new directory.
  local/thchs-30_data_prep.sh $H $thchs/data_thchs30 || exit 1;
  
  # Construct the phoneme-based lexicon from the CMU dicta
  # Create lexicon.txt, units.txt and lexicon_numbers.txt at data.dict_phn
  local/thchs30_prepare_phn_dict.sh || exit 1;
  
  # Compile the lexicon and token FSTs
  # generate lexicon FST L.fst according to words.txt, generate token FST T.fst according to tokens.txt
  ctc-crf/ctc_compile_dict_token.sh --dict-type "phn" \
    data/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;
  # Train and compile LMs. Generate G.fst according to lm, and compose FSTs into TLG.fst
  local/thchs30_train_lms.sh data/train/text data/dict_phn/lexicon.txt data/lm_phn || exit 1;
  local/thchs30_decode_graph.sh data/lm_phn data/lang_phn data/lang_phn_test || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "FBank Feature Generation"
  #perturb the speaking speed to achieve data augmentation
  utils/data/perturb_data_dir_speed_3way.sh data/train data/train_sp
  utils/data/perturb_data_dir_speed_3way.sh data/dev data/dev_sp
  echo " preparing directory for speed-perturbed data done"
  
  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  fbankdir=fbank
  for set in train_sp dev_sp; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 20 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;  #filter and sort the data files
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;  #achieve cmvn normalization
  done
  for set in test; do
    steps/make_fbank.sh --cmd "$train_cmd" --nj 20 data/$set exp/make_fbank/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;  #filter and sort the data files
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank/$set $fbankdir || exit 1;  #achieve cmvn normalization
  done
fi

data_tr=data/train_sp
data_cv=data/dev_sp

if [ $stage -le 3 ]; then
  #convert word sequences to label sequences according to lexicon_numbers.txt and text files in data/lang_phn
  #the result will be placed in $data_tr/ and $data_cv/
  utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_tr/text "<UNK>" > $data_tr/text_number
  utils/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_cv/text "<UNK>" > $data_cv/text_number
  echo "convert text_number finished"

  # prepare denominator
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train/text "<UNK>" > data/train/text_number
  #sort the text_number file, and then remove the duplicate lines
  cat data/train/text_number | sort -k 2 | uniq -f 1 > data/train/unique_text_number
  mkdir -p data/den_meta
  #generate phone_lm.fst, a phone-based language model
  chain-est-phone-lm ark:data/train/unique_text_number data/den_meta/phone_lm.fst
  #generate the correct T.fst, called T_den.fst
  ctc-crf/ctc_token_fst_corrected.py den data/lang_phn/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst
  #compose T_den.fst and phone_lm.fst into den_lm.fst
  fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst
  echo "prepare denominator finished"
  
fi
if [ $stage -le 4 ]; then 
  #calculate and save the weight for each label sequence based on text_number and phone_lm.fst
  path_weight $data_tr/text_number data/den_meta/phone_lm.fst > $data_tr/weight
  path_weight $data_cv/text_number data/den_meta/phone_lm.fst > $data_cv/weight
  echo "prepare weight finished"
  #apply CMVN feature normalization, calculate delta features, then sub-sample the input feature sequence
  feats_tr="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$data_tr/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
#echo "$feats_tr"
  feats_cv="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$data_cv/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
  ark_dir=data/all_ark
  mkdir -p $ark_dir
  #copy feature files, generate scp and ark files to save features.
  copy-feats "$feats_tr" "ark,scp:$ark_dir/tr.ark,$ark_dir/tr.scp" || exit 1
  copy-feats "$feats_cv" "ark,scp:$ark_dir/cv.ark,$ark_dir/cv.scp" || exit 1
  echo "copy -feats finished"
  ark_dir=data/all_ark
  mkdir -p data/hdf5
  #create a hdf5 file to save the feature, text_number and path weights.
  python3 ctc-crf/convert_to_hdf5.py $ark_dir/cv.scp $data_cv/text_number $data_cv/weight data/hdf5/cv.hdf5
  python3 ctc-crf/convert_to_hdf5.py $ark_dir/tr.scp $data_tr/text_number $data_tr/weight data/hdf5/tr.hdf5
fi

data_test=data/test

#do the same operations to test data as in stage 4.
if [ $stage -le 5 ]; then
  feats_test="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_test/utt2spk scp:$data_test/cmvn.scp scp:$data_test/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
  mkdir -p data/test_data
  copy-feats "$feats_test" "ark,scp:data/test_data/test.ark,data/test_data/test.scp"
fi

arch=BLSTM
dir=exp/$arch
output_unit=$(awk '{if ($1 == "#0")print $2 - 1 ;}' data/lang_phn/tokens.txt)

if [ $stage -le 6 ]; then
echo "nn training."
    #start training.
    python steps/train.py --output_unit=218 --lamb=0.01 --data_path=$dir
    python3 ctc-crf/train.py \
        --arch=$arch \
        --output_unit=$output_unit \
        --lamb=0.01 \
        --data_path \
        $data/hdf5 \
        $dir
fi

nj=20

if [ $stage -le 7 ]; then
  for set in test; do
    CUDA_VISIBLE_DEVICES=0 \
    ctc-crf/decode.sh --cmd "$decode_cmd" --nj 20 --acwt 1.0 \
      data/lang_phn_test data/$set data/${set}_data/test.scp $dir/decode
  done
fi
