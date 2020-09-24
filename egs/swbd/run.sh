#!/bin/bash

# Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
# Apache 2.0.
# This script implements CTC-CRF training on SwitchBoard dataset.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh

stage=1

swbd=/data/LDC97S62
fisher_dirs="/data/LDC2004T19/fe_03_p1_tran/ /data/LDC2005T19/fe_03_p2_tran/"
eval2000_dirs="/data/LDC2002S09/hub5e_00 /data/LDC2002T43"


if [ $stage -le 1 ]; then
  echo "Data Preparation and FST Construction"
  # Use the same data preparation script from Kaldi

  local/swbd1_data_download.sh $swbd
  local/swbd1_prepare_phn_dict.sh
  local/swbd1_data_prep.sh $swbd
  local/eval2000_data_prep.sh $eval2000_dirs

  # Compile the lexicon and token FSTs
  ctc-crf/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn || exit 1;

  # Train and compile LMs.
  local/swbd1_train_lms.sh data/local/train/text data/local/dict_phn/lexicon.txt data/local/lm $fisher_dirs || exit 1;


  # Compiles G for swbd trigram LM
  LM=data/local/lm/sw1.o3g.kn.gz
  srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
                         data/lang_phn $LM data/local/dict_phn/lexicon.txt data/lang_phn_sw1_tg
  old_lang=data/lang_phn
  new_lang=data/lang_phn_sw1_fsh_fg
  mkdir -p $new_lang
  cp -r $old_lang/* $new_lang
  
  unk=`grep "<unk>" $new_lang/words.txt | awk '{print $2}'`
  bos=`grep "<s>" $new_lang/words.txt | awk '{print $2}'`
  eos=`grep "</s>" $new_lang/words.txt | awk '{print $2}'`

  LM=data/local/lm/sw1_fsh.o4g.kn.gz
  arpa-to-const-arpa --bos-symbol=$bos \
    --eos-symbol=$eos --unk-symbol=$unk \
    "gunzip -c $LM | utils/map_arpa_lm.pl $new_lang/words.txt|"  $new_lang/G.carpa  || exit 1;

  # Compile the language-model FST and the final decoding graph TLG.fst

  langdir=data/lang_phn_sw1_tg
  fsttablecompose ${langdir}/L.fst $langdir/G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstarcsort --sort_type=ilabel > $langdir/LG.fst || exit 1;
  fsttablecompose ${langdir}/T.fst $langdir/LG.fst > $langdir/TLG.fst || exit 1;

fi

if [ $stage -le 2 ]; then
  echo "FBank Feature Generation"

  # Use the first 4k sentences as dev set, around 5 hours
  utils/subset_data_dir.sh --first data/train 4000 data/train_dev
  n=$[`cat data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last data/train $n data/train_nodev
  # Finally the full training set, around 286 hours
  utils/data/remove_dup_utts.sh 300 data/train_nodev data/train_nodup

  utils/data/perturb_data_dir_speed_3way.sh data/train_nodup data/train_nodup_sp
  utils/data/perturb_data_dir_speed_3way.sh data/train_dev data/train_dev_sp
  echo " preparing directory for speed-perturbed data done"

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  fbankdir=fbank
  steps/make_fbank.sh --cmd "$train_cmd" --nj 32 data/train_nodup_sp exp/make_fbank/train_nodup_sp $fbankdir || exit 1;
  utils/fix_data_dir.sh data/train_nodup_sp || exit 1;
  steps/compute_cmvn_stats.sh data/train_nodup_sp exp/make_fbank/train_nodup_sp $fbankdir || exit 1;

  steps/make_fbank.sh --cmd "$train_cmd" --nj 20 data/train_dev_sp exp/make_fbank/train_dev_sp $fbankdir || exit 1;
  utils/fix_data_dir.sh data/train_dev_sp || exit 1;
  steps/compute_cmvn_stats.sh data/train_dev_sp exp/make_fbank/train_dev_sp $fbankdir || exit 1;

  steps/make_fbank.sh --cmd "$train_cmd" --nj 10 data/eval2000 exp/make_fbank/eval2000 $fbankdir || exit 1;
  utils/fix_data_dir.sh data/eval2000 || exit 1;
  steps/compute_cmvn_stats.sh data/eval2000 exp/make_fbank/eval2000 $fbankdir || exit 1;
fi

data_tr=data/train_nodup_sp
data_cv=data/train_dev_sp

if [ $stage -le 3 ]; then
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_tr/text "<unk>" > $data_tr/text_number
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt $data_cv/text "<unk>" > $data_cv/text_number
  echo "convert text_number finished"
 
  # prepare denominator
  ctc-crf/prep_ctc_trans.py data/lang_phn/lexicon_numbers.txt data/train_nodup/text "<unk>" > data/train_nodup/text_number
  cat data/train_nodup/text_number | sort -k 2 | uniq -f 1 > data/train_nodup/unique_text_number
  mkdir -p data/den_meta
  chain-est-phone-lm ark:data/train_nodup/unique_text_number data/den_meta/phone_lm.fst
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
  python ctc-crf/convert_to_hdf5.py data/all_ark/cv.scp $data_cv/text_number $data_cv/weight data/hdf5/cv.hdf5 || exit 1
  python ctc-crf/convert_to_hdf5.py data/all_ark/tr.scp $data_tr/text_number $data_tr/weight data/hdf5/tr.hdf5 || exit 1
fi

arch=BLSTM
dir=exp/$arch

if [ $stage -le 5 ]; then
  python ctc-crf/train.py \
      --output_unit=46 \
      --arch=$arch \
      --lamb=0.01 \
      --data_path=data/hdf5 \
      $dir
fi

data_eval2000=data/eval2000
ark_dir=exp/decode_eval2000/ark

if [ $stage -le 6 ]; then
  feats_eval2000="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_eval2000/utt2spk scp:$data_eval2000/cmvn.scp scp:$data_eval2000/feats.scp ark:- \
       | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"

  mkdir data/test_data
  copy-feats "$feats_eval2000"   ark,scp:data/test_data/eval2000.ark,data/test_data/eval2000.scp
fi

if [ $stage -le 7 ]; then
  nj=20
  mkdir $dir/decode_eval2000_sw1_tg
  ctc-crf/decode.sh --stage 0 \
      --cmd "$decode_cmd" --nj $nj --acwt 1.0 \
      data/lang_phn_sw1_tg data/eval2000 data/test_data/eval2000.scp $dir/decode_eval2000_sw1_tg
  steps/lmrescore_const_arpa.sh --cmd "$cmd" data/lang_phn_sw1_{tg,fsh_fg} data/eval2000 $dir/decode_eval2000_{tg,fsh_fg}  || exit 1;
fi

